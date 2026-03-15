# EZ Car Rental – Thompson Sampling Starter Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.figsize"] = (8, 4.5)

BASE_DIR = Path(__file__).resolve().parent
JOURNEYS_PATH = BASE_DIR / "journeys.csv"
UTIL_PATH = BASE_DIR / "utilization.csv"

journeys = pd.read_csv(JOURNEYS_PATH)
util = pd.read_csv(UTIL_PATH)

print("Journeys shape:", journeys.shape)
print("Utilization shape:", util.shape)
print(journeys.head())
print(util.head())



# Basic cleaning + feature engineering

journeys["price"] = (
    journeys["Trip Sum Trip Price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

journeys["start_time"] = pd.to_datetime(journeys["Trip Start At Local Time"])
journeys["city"] = journeys["Car Parking Address City"]

def time_bucket(hour):
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"

journeys["time_of_day"] = journeys["start_time"].dt.hour.map(time_bucket)

car_city = (
    journeys.groupby("Car ID Hash")["city"]
    .agg(lambda s: s.mode().iloc[0])
    .rename("city")
)

util["timestamp"] = pd.to_datetime(util["Car Hourly Utilization Aggregated At Time"])
util["time_of_day"] = util["timestamp"].dt.hour.map(time_bucket)
util["utilization_rate"] = (
    util["Car Hourly Utilization Sum Utilized Minutes"] /
    util["Car Hourly Utilization Sum Available Minutes"].replace({0: np.nan})
).fillna(0)

util = util.join(car_city, on="Car ID Hash")

print("Cities in journeys:", journeys["city"].nunique())
print("Cities matched into utilization:", util["city"].nunique())
print("Share of utilization rows without a city match:", util["city"].isna().mean())



journey_state = (
    journeys.groupby(["city", "time_of_day"])
    .agg(
        trip_count=("Trip ID Hash", "size"),
        avg_price=("price", "mean"),
        med_price=("price", "median"),
    )
    .reset_index()
)

util_nonmissing = util.dropna(subset=["city"]).copy()
util_nonmissing["util_bin"] = pd.cut(
    util_nonmissing["utilization_rate"],
    bins=[-0.001, 0.33, 0.66, 1.001],
    labels=["low", "medium", "high"]
)

util_state = (
    util_nonmissing.groupby(["city", "time_of_day", "util_bin"], observed=True)
    .agg(
        avg_util=("utilization_rate", "mean"),
        hours=("Car ID Hash", "size")
    )
    .reset_index()
)

states = util_state.merge(journey_state, on=["city", "time_of_day"], how="left")
states["trip_count"] = states["trip_count"].fillna(0)
states["avg_price"] = states["avg_price"].fillna(journeys["price"].median())
states["med_price"] = states["med_price"].fillna(journeys["price"].median())

states["trip_score"] = states["trip_count"] / states["trip_count"].max()
states["util_score"] = states["avg_util"].fillna(0)
states["strength"] = 0.55 * states["trip_score"] + 0.45 * states["util_score"]
states["base_prob"] = 0.08 + 0.55 * states["strength"]
states["base_price"] = states["med_price"].clip(lower=10)

print("Number of states:", len(states))
print(states.head())



arms = ["low", "medium", "high"]
arm_to_idx = {a: i for i, a in enumerate(arms)}
price_multipliers = {"low": 0.85, "medium": 1.00, "high": 1.20}

rows = []
for state_id, row in states.iterrows():
    elasticity = 1 - row["strength"]  # weak states are more price-sensitive
    demand_multipliers = {
        "low": 1 + (0.12 + 0.08 * elasticity),
        "medium": 1.0,
        "high": 1 - (0.06 + 0.18 * elasticity),
    }

    for arm in arms:
        true_p = float(np.clip(row["base_prob"] * demand_multipliers[arm], 0.02, 0.95))
        actual_price = float(row["base_price"] * price_multipliers[arm])
        expected_revenue = true_p * actual_price

        rows.append({
            "state_id": state_id,
            "city": row["city"],
            "time_of_day": row["time_of_day"],
            "util_bin": row["util_bin"],
            "arm": arm,
            "true_p": true_p,
            "actual_price": actual_price,
            "expected_revenue": expected_revenue,
            "hours": row["hours"],
        })

state_arm = pd.DataFrame(rows)
print(state_arm.head())

true_p = np.zeros((len(states), len(arms)))
prices = np.zeros((len(states), len(arms)))
for _, r in state_arm.iterrows():
    s = int(r["state_id"])
    aidx = arm_to_idx[r["arm"]]
    true_p[s, aidx] = r["true_p"]
    prices[s, aidx] = r["actual_price"]

true_best_arm = np.argmax(true_p * prices, axis=1)
state_freq = states["hours"].values / states["hours"].sum()

best_arm_table = (
    state_arm.loc[state_arm.groupby("state_id")["expected_revenue"].idxmax(),
                  ["city", "time_of_day", "util_bin", "arm", "expected_revenue"]]
    .rename(columns={"arm": "true_best_arm", "expected_revenue": "true_best_expected_revenue"})
)

print(best_arm_table.head(10))



def run_thompson_sampling(n_steps=120_000, seed=42):
    rng = np.random.default_rng(seed)

    alpha = np.ones((len(states), len(arms)))
    beta = np.ones((len(states), len(arms)))

    rewards = []
    rmse_log = []
    policy_log = []

    for t in range(1, n_steps + 1):
        s = rng.choice(np.arange(len(states)), p=state_freq)

        theta = rng.beta(alpha[s], beta[s])
        sampled_revenue = theta * prices[s]
        aidx = int(np.argmax(sampled_revenue))

        rental = rng.random() < true_p[s, aidx]
        reward = prices[s, aidx] if rental else 0.0

        alpha[s, aidx] += 1 if rental else 0
        beta[s, aidx] += 0 if rental else 1
        rewards.append(reward)

        if t % 200 == 0 or t == 1:
            post_mean = alpha / (alpha + beta)
            rmse = np.sqrt(np.mean((post_mean - true_p) ** 2))

            learned_best = np.argmax(post_mean * prices, axis=1)
            unweighted_acc = np.mean(learned_best == true_best_arm)
            weighted_acc = np.sum((learned_best == true_best_arm) * state_freq)

            rmse_log.append((t, rmse, np.sum(rewards)))
            policy_log.append((t, unweighted_acc, weighted_acc))

    post_mean = alpha / (alpha + beta)
    learned_best = np.argmax(post_mean * prices, axis=1)

    return {
        "alpha": alpha,
        "beta": beta,
        "post_mean": post_mean,
        "rewards": np.array(rewards),
        "rmse_log": pd.DataFrame(rmse_log, columns=["step", "rmse", "cumulative_reward"]),
        "policy_log": pd.DataFrame(policy_log, columns=["step", "unweighted_accuracy", "weighted_accuracy"]),
        "learned_best": learned_best,
    }

results = run_thompson_sampling()
results["rmse_log"].tail()



# Charts

rmse_log = results["rmse_log"]
policy_log = results["policy_log"]

fig, ax = plt.subplots()
ax.plot(rmse_log["step"], rmse_log["rmse"])
ax.set_title("Posterior RMSE vs. True Rental Probabilities")
ax.set_xlabel("Training steps")
ax.set_ylabel("RMSE")
plt.show()

fig, ax = plt.subplots()
ax.plot(rmse_log["step"], rmse_log["cumulative_reward"])
ax.set_title("Cumulative Reward During Training")
ax.set_xlabel("Training steps")
ax.set_ylabel("Cumulative reward")
plt.show()

fig, ax = plt.subplots()
ax.plot(policy_log["step"], policy_log["weighted_accuracy"])
ax.set_title("Weighted Policy Accuracy")
ax.set_xlabel("Training steps")
ax.set_ylabel("Accuracy")
plt.show()



def simulate_policy(policy, n_steps=100_000, seed=123):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_steps):
        s = rng.choice(np.arange(len(states)), p=state_freq)
        if callable(policy):
            aidx = int(policy(s, rng))
        else:
            aidx = int(policy[s])

        rental = rng.random() < true_p[s, aidx]
        total += prices[s, aidx] if rental else 0.0
    return total / n_steps

learned_policy = results["learned_best"]
always_low = np.full(len(states), arm_to_idx["low"])
always_medium = np.full(len(states), arm_to_idx["medium"])
always_high = np.full(len(states), arm_to_idx["high"])
random_policy = lambda s, rng: rng.integers(0, len(arms))

comparison = pd.DataFrame({
    "policy": ["learned_ts", "always_low", "always_medium", "always_high", "random"],
    "avg_reward_per_step": [
        simulate_policy(learned_policy),
        simulate_policy(always_low),
        simulate_policy(always_medium),
        simulate_policy(always_high),
        simulate_policy(random_policy),
    ]
}).sort_values("avg_reward_per_step", ascending=False)

print(comparison)



policy_table = states[["city", "time_of_day", "util_bin"]].copy()
policy_table["learned_best_arm"] = [arms[i] for i in results["learned_best"]]
policy_table["true_best_arm"] = [arms[i] for i in true_best_arm]
policy_table["state_weight"] = state_freq

print(policy_table.head(20))

policy_table.to_csv("learned_policy_table.csv", index=False)
comparison.to_csv("policy_comparison.csv", index=False)

print("Saved learned_policy_table.csv and policy_comparison.csv")
