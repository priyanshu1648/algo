#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::vector;

// ---------- 1. static Almgren-Chriss trajectory ----------
struct ACConfig {
    double X = 100000.0;         // shares to sell
    double T = 1.0;              // time horizon (1 = one trading day)
    int    N = 60;               // number of intervals
    double sigma = 0.30;         // volatility
    double eta = 2.5e-6;         // temporary impact
    double gamma = 2.5e-7;       // permanent impact
    double risk_aversion = 5e-5; // lambda
};

struct Trajectory {
    vector<double> t;
    vector<double> shares_remaining;
    vector<double> trade_this_interval;
};

Trajectory almgren_chriss_trajectory(const ACConfig& cfg) {
    // x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
    // x_j is shares still left at t_j, trade size = x_{j-1} - x_j
    double tau = cfg.T / cfg.N;
    double eta_tilde = cfg.eta - 0.5 * cfg.gamma * tau;
    double kappa_tilde_sq = (cfg.risk_aversion * cfg.sigma * cfg.sigma) / std::max(eta_tilde, 1e-12);
    kappa_tilde_sq = std::max(kappa_tilde_sq, 0.0);

    double kappa;
    if (kappa_tilde_sq * tau * tau < 1e-8) {
        kappa = std::sqrt(kappa_tilde_sq);  // basically risk neutral, close to TWAP
    } else {
        double arg = kappa_tilde_sq * tau * tau / 2.0 + 1.0;
        kappa = std::acosh(arg) / tau;
    }

    Trajectory traj;
    traj.t.resize(cfg.N + 1);
    traj.shares_remaining.resize(cfg.N + 1);
    traj.trade_this_interval.assign(cfg.N + 1, 0.0);

    for (int i = 0; i <= cfg.N; i++) {
        // same as np.linspace(0, T, N + 1)
        traj.t[i] = cfg.T * i / cfg.N;
        if (kappa * cfg.T < 1e-8) {
            traj.shares_remaining[i] = cfg.X * (1.0 - traj.t[i] / cfg.T);  // straight line
        } else {
            traj.shares_remaining[i] = cfg.X * std::sinh(kappa * (cfg.T - traj.t[i])) / std::sinh(kappa * cfg.T);
        }
    }

    // shares traded in each interval (last entry stays 0)
    for (int i = 0; i < cfg.N; i++) {
        traj.trade_this_interval[i] = -(traj.shares_remaining[i + 1] - traj.shares_remaining[i]);
    }
    return traj;
}

struct CostVar {
    double expected_cost;
    double variance;
    double std_dev;
};

CostVar expected_cost_and_variance(const ACConfig& cfg, const Trajectory& traj) {
    // expected cost and variance of a given trajectory (AC formulas)
    double tau = cfg.T / cfg.N;

    double perm_impact_cost = 0.5 * cfg.gamma * cfg.X * cfg.X;
    double sum_n2 = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < cfg.N; i++) {
        sum_n2 += traj.trade_this_interval[i] * traj.trade_this_interval[i];
        sum_x2 += traj.shares_remaining[i] * traj.shares_remaining[i];
    }
    double temp_impact_cost = (tau > 0) ? cfg.eta / tau * sum_n2 : 0.0;
    double expected_cost = perm_impact_cost + temp_impact_cost;
    double variance = cfg.sigma * cfg.sigma * sum_x2 * tau;

    return {expected_cost, variance, std::sqrt(variance)};
}

// ---------- 2. TWAP baseline (equal slices) ----------
Trajectory twap_trajectory(const ACConfig& cfg) {
    Trajectory traj;
    traj.t.resize(cfg.N + 1);
    traj.shares_remaining.resize(cfg.N + 1);
    traj.trade_this_interval.assign(cfg.N + 1, 0.0);
    for (int i = 0; i <= cfg.N; i++) {
        traj.t[i] = cfg.T * i / cfg.N;
        traj.shares_remaining[i] = cfg.X * (1.0 - traj.t[i] / cfg.T);
    }
    for (int i = 0; i < cfg.N; i++) {
        traj.trade_this_interval[i] = -(traj.shares_remaining[i + 1] - traj.shares_remaining[i]);
    }
    return traj;
}

// ---------- 3. adaptive version ----------
struct SimConfig {
    unsigned long long seed = 11;
    double vol_regime_shift_prob = 0.05;   // chance of a vol jump each interval
    double vol_shift_multiplier = 2.2;     // size of the jump
    double vol_reversion = 0.15;           // how fast vol drifts back to base
    double vol_forecast_shrinkage = 0.15;  // weight on current vol vs base vol
                                           // in the forecast (spikes fade, so
                                           // we don't trust them fully)
};

struct Record {
    int interval;
    double t;
    double trade_qty;
    double realized_vol;
    double price;
    double exec_price;
    double shares_remaining;
    double cumulative_cost;
};

vector<Record> simulate_adaptive_execution(const ACConfig& cfg, const SimConfig& sim_cfg, bool adaptive) {
    // Runs the order interval by interval against a price with changing
    // volatility. adaptive=true re-solves AC every step with an updated
    // vol estimate, adaptive=false just follows the t=0 schedule.
    std::mt19937_64 rng(sim_cfg.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    double tau = cfg.T / cfg.N;
    double base_vol = cfg.sigma;

    double arrival_price = 100.0;  // price when we decided to trade, used as benchmark
    double price = arrival_price;
    double realized_vol = base_vol;
    double shares_remaining = cfg.X;
    double total_cost = 0.0;
    vector<Record> records;
    records.reserve(cfg.N);

    Trajectory static_traj = almgren_chriss_trajectory(cfg);
    const vector<double>& static_trades = static_traj.trade_this_interval;

    for (int j = 0; j < cfg.N; j++) {
        // vol either jumps or reverts towards base (algo can't see this directly)
        if (uni(rng) < sim_cfg.vol_regime_shift_prob) {
            realized_vol *= sim_cfg.vol_shift_multiplier;
        } else {
            realized_vol += sim_cfg.vol_reversion * (base_vol - realized_vol);
        }
        realized_vol = std::max(realized_vol, 1e-4);

        double trade_qty;
        if (adaptive) {
            double remaining_time = cfg.T - j * tau;
            int remaining_steps = cfg.N - j;
            // blend current vol with base vol, otherwise a single spike
            // makes it trade way too fast
            double vol_forecast = sim_cfg.vol_forecast_shrinkage * realized_vol
                                + (1.0 - sim_cfg.vol_forecast_shrinkage) * base_vol;
            ACConfig sub_cfg;
            sub_cfg.X = shares_remaining;
            sub_cfg.T = remaining_time;
            sub_cfg.N = std::max(remaining_steps, 1);
            sub_cfg.sigma = vol_forecast;
            sub_cfg.eta = cfg.eta;
            sub_cfg.gamma = cfg.gamma;
            sub_cfg.risk_aversion = cfg.risk_aversion;
            Trajectory sub_traj = almgren_chriss_trajectory(sub_cfg);
            trade_qty = sub_traj.trade_this_interval[0];
        } else {
            trade_qty = static_trades[j];
        }

        trade_qty = std::min(trade_qty, shares_remaining);

        // price shock from volatility + impact from our own trade
        double price_shock = realized_vol * std::sqrt(tau) * normal(rng);
        double temp_impact = (tau > 0) ? cfg.eta * (trade_qty / tau) : 0.0;
        double perm_impact = cfg.gamma * trade_qty;

        double exec_price = price + temp_impact + 0.5 * perm_impact;
        // shortfall = what we paid vs the arrival price
        double cost_this_interval = trade_qty * (exec_price - arrival_price);
        total_cost += cost_this_interval;

        price = price + price_shock + perm_impact;
        shares_remaining -= trade_qty;

        records.push_back({j, j * tau, trade_qty, realized_vol, price,
                           exec_price, shares_remaining, total_cost});
    }
    return records;
}

struct Summary {
    std::string label;
    double total_cost;
    double cost_bps;
    double cost_path_volatility;
    double max_realized_vol;
};

// sample standard deviation (n - 1), same as pandas .std()
double sample_std(const vector<double>& v) {
    if (v.size() < 2) return NAN;
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double ss = 0.0;
    for (double x : v) ss += (x - m) * (x - m);
    return std::sqrt(ss / (v.size() - 1));
}

double mean_of(const vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// quantile with linear interpolation, same default as pandas
double quantile(vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    size_t hi = (size_t)std::ceil(pos);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

Summary summarize_execution(const vector<Record>& result, const ACConfig& cfg, const std::string& label) {
    double total_shares = cfg.X;
    double total_cost = result.back().cumulative_cost;
    double cost_bps = (total_cost / (total_shares * 100.0)) * 1e4;  // bps of notional (price = 100)

    // std of the per-interval changes in cumulative cost
    vector<double> diffs;
    for (size_t i = 1; i < result.size(); i++) {
        diffs.push_back(result[i].cumulative_cost - result[i - 1].cumulative_cost);
    }
    double vol_of_cost_path = sample_std(diffs);

    double max_vol = 0.0;
    for (const auto& r : result) max_vol = std::max(max_vol, r.realized_vol);

    return {label, total_cost, cost_bps, vol_of_cost_path, max_vol};
}

void write_trials_csv(const std::string& path, const vector<Summary>& rows) {
    std::ofstream f(path);
    f << "label,total_cost,cost_bps,cost_path_volatility,max_realized_vol\n";
    for (const auto& s : rows) {
        f << s.label << "," << s.total_cost << "," << s.cost_bps << ","
          << s.cost_path_volatility << "," << s.max_realized_vol << "\n";
    }
}

void write_path_csv(const std::string& path, const vector<Record>& rows) {
    std::ofstream f(path);
    f << "interval,t,trade_qty,realized_vol,price,exec_price,shares_remaining,cumulative_cost\n";
    for (const auto& r : rows) {
        f << r.interval << "," << r.t << "," << r.trade_qty << "," << r.realized_vol << ","
          << r.price << "," << r.exec_price << "," << r.shares_remaining << ","
          << r.cumulative_cost << "\n";
    }
}

void print_stats(const char* title, const vector<double>& bps) {
    std::printf("%s\n", title);
    std::printf("mean  %.6f\n", mean_of(bps));
    std::printf("std   %.6f\n", sample_std(bps));
    std::printf("50%%   %.6f\n\n", quantile(bps, 0.5));
}

int main() {
    ACConfig ac_cfg;

    // run many random scenarios, same seed for static and adaptive in each
    int n_trials = 300;
    vector<Summary> static_results, adaptive_results;
    for (int trial = 0; trial < n_trials; trial++) {
        SimConfig sim_cfg_trial;
        sim_cfg_trial.seed = trial;
        auto static_res = simulate_adaptive_execution(ac_cfg, sim_cfg_trial, false);
        auto adaptive_res = simulate_adaptive_execution(ac_cfg, sim_cfg_trial, true);
        static_results.push_back(summarize_execution(static_res, ac_cfg, "Static AC"));
        adaptive_results.push_back(summarize_execution(adaptive_res, ac_cfg, "Adaptive AC"));
    }

    vector<double> static_bps, adaptive_bps;
    for (const auto& s : static_results) static_bps.push_back(s.cost_bps);
    for (const auto& s : adaptive_results) adaptive_bps.push_back(s.cost_bps);

    print_stats("=== Static AC ===", static_bps);
    print_stats("=== Adaptive AC ===", adaptive_bps);

    double static_std = sample_std(static_bps);
    double adaptive_std = sample_std(adaptive_bps);

    std::printf("Mean cost change: %+.3f bps\n", mean_of(adaptive_bps) - mean_of(static_bps));
    std::printf("Cost std (static): %.3f bps\n", static_std);
    std::printf("Cost std (adaptive): %.3f bps\n", adaptive_std);
    std::printf("Cost std reduction: %.1f%%\n", (1.0 - adaptive_std / static_std) * 100.0);

    double static_p95 = quantile(static_bps, 0.95);
    double adaptive_p95 = quantile(adaptive_bps, 0.95);
    std::printf("95th percentile cost (static): %.3f bps\n", static_p95);
    std::printf("95th percentile cost (adaptive): %.3f bps\n", adaptive_p95);
    std::printf("p95 cost reduction: %.1f%%\n", (1.0 - adaptive_p95 / static_p95) * 100.0);

    write_trials_csv("static_ac_trials.csv", static_results);
    write_trials_csv("adaptive_ac_trials.csv", adaptive_results);

    // one example path each, for plotting later
    SimConfig rep_cfg;
    rep_cfg.seed = 3;
    write_path_csv("rep_static_path.csv", simulate_adaptive_execution(ac_cfg, rep_cfg, false));
    write_path_csv("rep_adaptive_path.csv", simulate_adaptive_execution(ac_cfg, rep_cfg, true));

    return 0;
}