#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {

struct NativeStats {
    double learner_loss;
    double short_learner_loss;
    double long_learner_loss;
    double maximum_weighted_gradient_norm;
    int64_t number_nonfinite_updates;
    int64_t maximum_active;
    int64_t saturated_rounds;
    int64_t observed_short;
    int64_t observed_long;
    int64_t weight_count;
};


static constexpr uint8_t ACTIVE = 0;
static constexpr uint8_t OBSERVED = 1;
static constexpr uint8_t PROXY_REJECTED = 2;
static constexpr uint8_t PROXY_EXPIRED = 3;
static constexpr uint8_t CAPACITY_REJECTED = 4;
static constexpr uint8_t HORIZON_UNRESOLVED = 5;

int simulate_delayed(
    int64_t horizon,
    int64_t dimension,
    int64_t capacity,
    int problem,
    int scheduler,
    const int64_t* delays,
    const uint8_t* is_long,
    const double* short_vector,
    const double* long_vector,
    const double* explicit_gradients,
    const float* directions,
    const int64_t* proxy_draws,
    double pareto_shape,
    double pareto_scale,
    double uniform_probability,
    double loss_offset,
    double quadratic_coefficient,
    double radius,
    double delta,
    double eta,
    uint8_t* states,
    double* retained_weights,
    NativeStats* stats
) {
    if (horizon <= 0 || dimension < 2 || capacity <= 0 || !stats) return 1;
    std::fill(states, states + horizon, uint8_t(255));
    std::vector<int64_t> arrival_head(horizon, -1), arrival_next(horizon, -1);
    std::vector<int64_t> expiry_head(horizon, -1), expiry_next(horizon, -1);


    for (int64_t s = horizon - 1; s >= 0; --s) {
        if (delays[s] > 0) {
            const int64_t arrival = s + delays[s];
            if (arrival < 0) return 2;
            if (arrival < horizon) {
                arrival_next[s] = arrival_head[arrival];
                arrival_head[arrival] = s;
            }
        }
    }
    std::vector<double> point(dimension, 0.0);
    std::vector<double> update(dimension, 0.0);
    std::vector<double> scalar_losses(problem == 1 ? horizon : 0, 0.0);
    std::vector<double> played_gradients(
        problem == 0 && quadratic_coefficient != 0.0 ? horizon * dimension : 0, 0.0);
    int64_t active_count = 0;
    *stats = NativeStats{};

    auto gradient_value = [&](int64_t round, int64_t coordinate) -> double {
        if (explicit_gradients) return explicit_gradients[round * dimension + coordinate];
        return is_long[round] ? long_vector[coordinate] : short_vector[coordinate];
    };
    auto importance_weight = [&](int64_t delay_value) -> double {
        if (scheduler == 0 || delay_value == 0) return 1.0;
        if (scheduler == 1) {
            const double q = std::pow(1.0 + double(delay_value) / pareto_scale, -pareto_shape);
            return 1.0 / q;
        }
        return 1.0 / uniform_probability;
    };
    const double internal_radius = problem == 0 ? radius : radius - delta;
    auto project_point = [&]() {
        double squared_norm = 0.0;
        for (double coordinate : point) squared_norm += coordinate * coordinate;
        const double norm = std::sqrt(squared_norm);
        if (norm > internal_radius && norm > 0.0) {
            const double factor = internal_radius / norm;
            for (double& coordinate : point) coordinate *= factor;
        }
    };
    auto observe = [&](int64_t round) {
        const double weight = importance_weight(delays[round]);
        retained_weights[stats->weight_count++] = weight;
        if (is_long[round]) ++stats->observed_long;
        else ++stats->observed_short;
        double squared_norm = 0.0;
        bool finite = std::isfinite(weight);
        if (problem == 0) {
            for (int64_t j = 0; j < dimension; ++j) {
                const double gradient = played_gradients.empty()
                    ? gradient_value(round, j) : played_gradients[round * dimension + j];
                update[j] = weight * gradient;
                squared_norm += update[j] * update[j];
                finite = finite && std::isfinite(update[j]);
            }
        } else {


            const double multiplier = weight * (double(dimension) / delta)
                * scalar_losses[round];
            for (int64_t j = 0; j < dimension; ++j) {
                update[j] = multiplier * double(directions[round * dimension + j]);
                squared_norm += update[j] * update[j];
                finite = finite && std::isfinite(update[j]);
            }
        }
        if (finite && std::isfinite(squared_norm)) {
            for (int64_t j = 0; j < dimension; ++j)
                point[j] -= eta * update[j];
            project_point();
            stats->maximum_weighted_gradient_norm = std::max(
                stats->maximum_weighted_gradient_norm, std::sqrt(squared_norm));
        } else {
            ++stats->number_nonfinite_updates;
        }
        states[round] = OBSERVED;
    };

    std::vector<double> action(dimension, 0.0);
    for (int64_t time = 0; time < horizon; ++time) {

        for (int64_t round = arrival_head[time]; round >= 0; round = arrival_next[round]) {
            if (states[round] == ACTIVE) {
                --active_count;
                observe(round);
            }
        }
        for (int64_t round = expiry_head[time]; round >= 0; round = expiry_next[round]) {
            if (states[round] == ACTIVE) {
                --active_count;
                states[round] = PROXY_EXPIRED;
            }
        }

        for (int64_t j = 0; j < dimension; ++j) action[j] = point[j];
        if (problem == 1) {
            for (int64_t j = 0; j < dimension; ++j)
                action[j] += delta * double(directions[time * dimension + j]);
        }
        double loss = loss_offset;
        for (int64_t j = 0; j < dimension; ++j)
            loss += gradient_value(time, j) * action[j];
        if (quadratic_coefficient != 0.0) {
            double action_norm_squared = 0.0;
            for (int64_t j = 0; j < dimension; ++j) {
                action_norm_squared += action[j] * action[j];
                if (!played_gradients.empty())
                    played_gradients[time * dimension + j] = gradient_value(time, j)
                        + 2.0 * quadratic_coefficient * action[j];
            }
            loss += quadratic_coefficient * action_norm_squared;
            if (!(loss >= -1e-7 && loss <= 1.0 + 1e-7)) return 5;
        }
        stats->learner_loss += loss;
        if (is_long[time]) stats->long_learner_loss += loss;
        else stats->short_learner_loss += loss;
        if (problem == 1) scalar_losses[time] = loss;

        if (scheduler != 0 && active_count >= capacity) ++stats->saturated_rounds;
        if (delays[time] == 0) {
            observe(time);
            continue;
        }
        const int64_t lifetime = proxy_draws[time];
        if (lifetime == 0) {
            states[time] = PROXY_REJECTED;
        } else if (scheduler != 0 && active_count >= capacity) {
            states[time] = CAPACITY_REJECTED;
        } else {
            states[time] = ACTIVE;
            ++active_count;
            stats->maximum_active = std::max(stats->maximum_active, active_count);
            if (scheduler != 0 && active_count > capacity) return 3;
            if (lifetime > 0) {
                const int64_t expiry = time + lifetime;
                if (expiry < horizon) {
                    expiry_next[time] = expiry_head[expiry];
                    expiry_head[expiry] = time;
                }
            }
        }
    }
    for (int64_t round = 0; round < horizon; ++round) {
        if (states[round] == ACTIVE) states[round] = HORIZON_UNRESOLVED;
        if (states[round] == uint8_t(255)) return 4;
    }
    return 0;
}

}
