"""
Training runner.
Run once before the Streamlit demo:  python train.py
Saves trained weights + experiment results to tasteflow_results.pkl
"""

import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from env import TasteFlowEnv, UserProfile, CATEGORIES, MEALS
from agents import RandomAgent, RuleBasedAgent, LinUCBAgent, PPOAgent, CPOAgent, GreedyAgent, EpsilonGreedyAgent

N_TRAIN_EPISODES = 800
N_EVAL_EPISODES  = 50
PRINT_EVERY      = 100
WEEKLY_BUDGET    = 200
WEEKLY_CALORIES  = 18000


def make_profile(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return UserProfile(
        preferred_cat="Japanese", budget_sensitivity=0.5,
        calorie_sensitivity=0.3, fatigue_sensitivity=0.6,
        noise=0.1, preference_drift_at=10, drift_to="Korean",
    )


def run_ppo(env, agent, train=True):
    s = env.reset(); done = False; total = 0.0
    while not done:
        valid = env.get_valid_actions()
        a = agent.select(s, valid)
        lp = agent.logprob(s, a, valid) if train else 0.0
        ns, r, done, info = env.step(a)
        if train:
            agent.store(s, a, r, lp, float(done), valid)
        s = ns; total += r
    if train:
        agent.update()
    return total, env.summary()


def run_cpo(env, agent, train=True):
    s = env.reset(); done = False; total = 0.0
    while not done:
        valid = env.get_valid_actions()
        a = agent.select(s, valid)
        lp = agent.logprob(s, a, valid) if train else 0.0
        ns, r, done, info = env.step(a)
        if train:
            r_accept = info.get("r_accept", r)
            b_cost = MEALS[a]["price"]    / WEEKLY_BUDGET
            c_cost = MEALS[a]["calories"] / WEEKLY_CALORIES
            agent.store(s, a, r_accept, lp, float(done), valid, b_cost, c_cost)
        s = ns; total += r
    if train:
        summ = env.summary()
        agent.update(
            episode_budget_fraction  = summ["budget_spent"]   / WEEKLY_BUDGET,
            episode_calorie_fraction = summ["calories_eaten"] / WEEKLY_CALORIES,
        )
    return total, env.summary()


def run_generic(env, agent, train=True):
    s = env.reset(); done = False; total = 0.0
    while not done:
        valid = env.get_valid_actions()
        a = agent.select(s, valid)
        ns, r, done, info = env.step(a)
        if train and hasattr(agent, "update"):
            agent.update(a, s, r)
        s = ns; total += r
    return total, env.summary()


def train_and_evaluate():
    print("=" * 60)
    print("  TasteFlow — Training Run  (PPO + CPO + Baselines)")
    print("=" * 60)

    # ── Train PPO ──────────────────────────────────────────────────
    print("\n  [1/2] Training PPO...")
    ppo = PPOAgent(lr=5e-3, gamma=0.95)
    ppo_curve, recent = [], []
    for ep in range(1, N_TRAIN_EPISODES + 1):
        env = TasteFlowEnv(WEEKLY_BUDGET, WEEKLY_CALORIES, make_profile(ep))
        r, _ = run_ppo(env, ppo)
        recent.append(r)
        if len(recent) > 20: recent.pop(0)
        if ep % 10 == 0:
            ppo_curve.append((ep, float(np.mean(recent))))
        if ep % PRINT_EVERY == 0:
            print(f"    ep {ep:4d} | Mean(20): {np.mean(recent):+.1f}")

    # ── Train CPO ──────────────────────────────────────────────────
    print("\n  [2/2] Training CPO...")
    cpo = CPOAgent(lr_policy=3e-3, lr_lambda=8e-3, gamma=0.95)
    cpo_curve, lambda_b_curve, lambda_c_curve, recent = [], [], [], []
    for ep in range(1, N_TRAIN_EPISODES + 1):
        env = TasteFlowEnv(WEEKLY_BUDGET, WEEKLY_CALORIES, make_profile(ep))
        r, _ = run_cpo(env, cpo)
        recent.append(r)
        if len(recent) > 20: recent.pop(0)
        if ep % 10 == 0:
            cpo_curve.append((ep, float(np.mean(recent))))
            lambda_b_curve.append((ep, cpo.lambda_budget))
            lambda_c_curve.append((ep, cpo.lambda_calorie))
        if ep % PRINT_EVERY == 0:
            print(f"    ep {ep:4d} | Mean(20): {np.mean(recent):+.1f} | "
                  f"λ_b={cpo.lambda_budget:.3f} λ_c={cpo.lambda_calorie:.3f}")

    # ── Evaluate all agents ────────────────────────────────────────
    print("\n  Evaluating...")
    agent_configs = [
        ("Random",      RandomAgent(),         run_generic),
        ("Greedy",      GreedyAgent(),         run_generic),
        ("ε-Greedy",    EpsilonGreedyAgent(),  run_generic),
        ("Rule-Based",  RuleBasedAgent(),      run_generic),
        ("LinUCB",      LinUCBAgent(),         run_generic),
        ("PPO",         ppo,                   run_ppo),
        ("CPO",         cpo,                   run_cpo),
    ]
    results = {}
    for name, agent, run_fn in agent_configs:
        rews, bok, cok, churns, bfrac, cfrac = [], [], [], [], [], []
        for i in range(N_EVAL_EPISODES):
            env = TasteFlowEnv(WEEKLY_BUDGET, WEEKLY_CALORIES, make_profile(1000+i))
            r, s = run_fn(env, agent, train=False)
            rews.append(r);  bok.append(int(s["budget_ok"]))
            cok.append(int(s["calorie_ok"])); churns.append(s["churns"])
            bfrac.append(s["budget_spent"]/WEEKLY_BUDGET)
            cfrac.append(s["calories_eaten"]/WEEKLY_CALORIES)
        results[name] = {
            "mean_reward":  float(np.mean(rews)),
            "std_reward":   float(np.std(rews)),
            "budget_pct":   float(np.mean(bok)  * 100),
            "calorie_pct":  float(np.mean(cok)  * 100),
            "churn_rate":   float(np.mean(churns) / 21 * 100),
            "mean_b_frac":  float(np.mean(bfrac)),
            "mean_c_frac":  float(np.mean(cfrac)),
            "constraint_ok_pct": float(np.mean(
                [int(b and c) for b, c in zip(bok, cok)]) * 100),
        }
        r = results[name]
        print(f"  {name:12s} | R:{r['mean_reward']:+.1f}±{r['std_reward']:.1f} | "
              f"Budget:{r['budget_pct']:.0f}% Cal:{r['calorie_pct']:.0f}% "
              f"Both:{r['constraint_ok_pct']:.0f}% Churn:{r['churn_rate']:.1f}%")

    # ── Collect demo logs ──────────────────────────────────────────
    def demo_log(agent, run_fn):
        env = TasteFlowEnv(WEEKLY_BUDGET, WEEKLY_CALORIES,
                           UserProfile("Japanese", fatigue_sensitivity=0.6, noise=0.05))
        run_fn(env, agent, train=False)
        return env.log

    log_random   = demo_log(RandomAgent(),         run_generic)
    log_greedy   = demo_log(GreedyAgent(),         run_generic)
    log_egreedy  = demo_log(EpsilonGreedyAgent(),  run_generic)
    log_ppo      = demo_log(ppo,                   run_ppo)
    log_cpo      = demo_log(cpo,                   run_cpo)

    # ── Save ──────────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(__file__), "tasteflow_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "ppo_curve": ppo_curve, "cpo_curve": cpo_curve,
            "lambda_b_curve": lambda_b_curve, "lambda_c_curve": lambda_c_curve,
            "results": results,
            "log_before":   log_random,
            "log_greedy":   log_greedy,
            "log_egreedy":  log_egreedy,
            "log_after":    log_ppo,
            "log_cpo":      log_cpo,
            "ppo_weights": {k: getattr(ppo, k)
                for k in ["W1","b1","W2","b2","Wp","bp","Wv","bv"]},
            "cpo_weights": {k: getattr(cpo, k)
                for k in ["W1","b1","W2","b2","Wp","bp","Wv","bv",
                          "lambda_budget","lambda_calorie"]},
        }, f)
    print(f"\n  Saved → {save_path}")
    return ppo_curve, cpo_curve, results


if __name__ == "__main__":
    train_and_evaluate()
