"""
Training runner.

Usage:
  python train.py                # full run: train all agents + evaluate + save
  python train.py --ablation     # additional ablation study (env components)

Saves trained weights + experiment results to tasteflow_results.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env import TasteFlowEnv, UserProfile, MEALS  # noqa: E402
from agents import (                                # noqa: E402
    RandomAgent, RuleBasedAgent, LinUCBAgent,
    PPOAgent, LagrangianPPOAgent, GreedyAgent, EpsilonGreedyAgent,
)

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


def make_env(profile, **env_kwargs):
    return TasteFlowEnv(WEEKLY_BUDGET, WEEKLY_CALORIES, profile, **env_kwargs)


# ── Episode runners ──────────────────────────────────────────────────────
def run_ppo(env, agent, train=True):
    s = env.reset(); done = False; total = 0.0
    while not done:
        valid = env.get_valid_actions()
        a  = agent.select(s, valid)
        lp = agent.logprob(s, a, valid) if train else 0.0
        ns, r, done, info = env.step(a)
        if train:
            agent.store(s, a, r, lp, float(done), valid)
        s = ns; total += r
    if train:
        agent.update()
    return total, env.summary()


def run_ppol(env, agent, train=True):
    s = env.reset(); done = False; total = 0.0
    while not done:
        valid = env.get_valid_actions()
        a  = agent.select(s, valid)
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


# ── Helpers ──────────────────────────────────────────────────────────────
def ci95(values):
    """95% confidence interval half-width using normal approximation."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    return 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr))


def train_baseline(name, agent, run_fn, n_episodes, env_kwargs=None):
    """Pre-train a baseline (Greedy / ε-Greedy / LinUCB / Rule-Based) with
    the same episode budget as PPO/CPO so eval comparisons are fair."""
    env_kwargs = env_kwargs or {}
    rolling = []
    for ep in range(1, n_episodes + 1):
        env = make_env(make_profile(ep), **env_kwargs)
        r, _ = run_fn(env, agent, train=True)
        rolling.append(r)
        if len(rolling) > 50: rolling.pop(0)
        if ep % PRINT_EVERY == 0:
            print(f"    {name:12s} ep {ep:4d} | Mean(50): {np.mean(rolling):+.1f}")


def evaluate(name, agent, run_fn, n_eval, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    rews, bok, cok, churns, bfrac, cfrac = [], [], [], [], [], []
    for i in range(n_eval):
        env = make_env(make_profile(1000 + i), **env_kwargs)
        r, s = run_fn(env, agent, train=False)
        rews.append(r);  bok.append(int(s["budget_ok"]))
        cok.append(int(s["calorie_ok"])); churns.append(s["churns"])
        bfrac.append(s["budget_spent"]   / WEEKLY_BUDGET)
        cfrac.append(s["calories_eaten"] / WEEKLY_CALORIES)
    out = {
        "mean_reward":       float(np.mean(rews)),
        "std_reward":        float(np.std(rews)),
        "ci95_reward":       ci95(rews),
        "budget_pct":        float(np.mean(bok)  * 100),
        "calorie_pct":       float(np.mean(cok)  * 100),
        "churn_rate":        float(np.mean(churns) / 21 * 100),
        "mean_b_frac":       float(np.mean(bfrac)),
        "mean_c_frac":       float(np.mean(cfrac)),
        "constraint_ok_pct": float(np.mean(
            [int(b and c) for b, c in zip(bok, cok)]) * 100),
    }
    print(f"  {name:14s} | R:{out['mean_reward']:+.1f}±{out['ci95_reward']:.1f}(95%CI) | "
          f"Budget:{out['budget_pct']:>3.0f}% Cal:{out['calorie_pct']:>3.0f}% "
          f"Both:{out['constraint_ok_pct']:>3.0f}% Churn:{out['churn_rate']:>4.1f}%")
    return out


# ── Main training & evaluation pipeline ─────────────────────────────────
def train_and_evaluate():
    print("=" * 64)
    print("  TasteFlow — Training Run  (PPO + PPO-L + fair-baseline comparison)")
    print(f"  Train episodes: {N_TRAIN_EPISODES}   Eval episodes: {N_EVAL_EPISODES}")
    print("=" * 64)

    # ── Train PPO ──────────────────────────────────────────────────
    print("\n  [1/4] Training PPO...")
    ppo = PPOAgent(lr=5e-3, gamma=0.95, n_episodes_per_update=4)
    ppo_curve, recent = [], []
    for ep in range(1, N_TRAIN_EPISODES + 1):
        env = make_env(make_profile(ep))
        r, _ = run_ppo(env, ppo)
        recent.append(r)
        if len(recent) > 20: recent.pop(0)
        if ep % 10 == 0:
            ppo_curve.append((ep, float(np.mean(recent))))
        if ep % PRINT_EVERY == 0:
            print(f"    PPO ep {ep:4d} | Mean(20): {np.mean(recent):+.1f}")

    # ── Train PPO-Lagrangian (PPO-L) ───────────────────────────────
    print("\n  [2/4] Training PPO-Lagrangian (PPO-L)...")
    ppol = LagrangianPPOAgent(
        lr_policy=3e-3, lr_lambda=1e-1, gamma=0.95,
        lambda_init=0.5, lambda_max=50.0,
        n_episodes_per_update=1,   # per-episode policy update so dual & primal stay in sync
    )
    ppol_curve, lambda_b_curve, lambda_c_curve, recent = [], [], [], []
    for ep in range(1, N_TRAIN_EPISODES + 1):
        env = make_env(make_profile(ep))
        r, _ = run_ppol(env, ppol)
        recent.append(r)
        if len(recent) > 20: recent.pop(0)
        if ep % 10 == 0:
            ppol_curve.append((ep, float(np.mean(recent))))
            lambda_b_curve.append((ep, ppol.lambda_budget))
            lambda_c_curve.append((ep, ppol.lambda_calorie))
        if ep % PRINT_EVERY == 0:
            print(f"    PPO-L ep {ep:4d} | Mean(20): {np.mean(recent):+.1f} | "
                  f"λ_b={ppol.lambda_budget:.3f} λ_c={ppol.lambda_calorie:.3f}")

    # ── Pre-train baselines for fair comparison ────────────────────
    # Greedy / ε-Greedy / LinUCB get the SAME N_TRAIN_EPISODES budget
    # before evaluation. Frozen (no update) during eval.
    print("\n  [3/4] Pre-training baselines for fair comparison...")
    greedy   = GreedyAgent()
    egreedy  = EpsilonGreedyAgent(epsilon=0.1)
    linucb   = LinUCBAgent(alpha=0.5)
    rulebased= RuleBasedAgent()
    train_baseline("Greedy",   greedy,   run_generic, N_TRAIN_EPISODES)
    train_baseline("εGreedy",  egreedy,  run_generic, N_TRAIN_EPISODES)
    train_baseline("LinUCB",   linucb,   run_generic, N_TRAIN_EPISODES)
    # Rule-based has no learning, but we still expose it to env for visit_counts

    # ── Evaluate all agents ────────────────────────────────────────
    print("\n  [4/4] Evaluating all agents (95% CI shown)...")
    agent_configs = [
        ("Random",      RandomAgent(),    run_generic),
        ("Greedy",      greedy,           run_generic),
        ("ε-Greedy",    egreedy,          run_generic),
        ("Rule-Based",  rulebased,        run_generic),
        ("LinUCB",      linucb,           run_generic),
        ("PPO",         ppo,              run_ppo),
        ("PPO-L",       ppol,             run_ppol),
    ]
    results = {}
    for name, agent, run_fn in agent_configs:
        results[name] = evaluate(name, agent, run_fn, N_EVAL_EPISODES)

    # ── Collect demo logs ──────────────────────────────────────────
    def demo_log(agent, run_fn):
        env = make_env(UserProfile("Japanese", fatigue_sensitivity=0.6, noise=0.05))
        run_fn(env, agent, train=False)
        return env.log

    log_random  = demo_log(RandomAgent(),  run_generic)
    log_greedy  = demo_log(greedy,         run_generic)
    log_egreedy = demo_log(egreedy,        run_generic)
    log_ppo     = demo_log(ppo,            run_ppo)
    log_ppol    = demo_log(ppol,           run_ppol)

    # ── Save ──────────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(__file__), "tasteflow_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "ppo_curve":      ppo_curve,
            "ppol_curve":     ppol_curve,
            "lambda_b_curve": lambda_b_curve,
            "lambda_c_curve": lambda_c_curve,
            "results":        results,
            "log_before":     log_random,
            "log_greedy":     log_greedy,
            "log_egreedy":    log_egreedy,
            "log_after":      log_ppo,
            "log_ppol":       log_ppol,
            "ppo_weights":    {k: getattr(ppo, k)
                               for k in ["W1","b1","W2","b2","Wp","bp","Wv","bv"]},
            "ppol_weights":   {k: getattr(ppol, k)
                               for k in ["W1","b1","W2","b2","Wp","bp","Wv","bv",
                                         "lambda_budget","lambda_calorie"]},
            "config": {
                "n_train_episodes":      N_TRAIN_EPISODES,
                "n_eval_episodes":       N_EVAL_EPISODES,
                "weekly_budget":         WEEKLY_BUDGET,
                "weekly_calories":       WEEKLY_CALORIES,
                "n_episodes_per_update": 4,
                "fair_baselines":        True,
            },
        }, f)
    print(f"\n  Saved → {save_path}")
    return ppo_curve, ppol_curve, results


# ── Ablation study ──────────────────────────────────────────────────────
def ablation_study():
    """
    Trains a fresh PPO agent under several env-component ablations and
    reports the eval metrics.  Demonstrates which reward components matter.
    """
    print("\n" + "=" * 64)
    print("  TasteFlow — Ablation Study  (PPO under env-component flips)")
    print("=" * 64)

    ablations = [
        ("Full",            dict(use_r_goal=True,  use_terminal_reward=True,  use_fatigue_dynamics=True)),
        ("No r_goal",       dict(use_r_goal=False, use_terminal_reward=True,  use_fatigue_dynamics=True)),
        ("No terminal",     dict(use_r_goal=True,  use_terminal_reward=False, use_fatigue_dynamics=True)),
        ("No fatigue",      dict(use_r_goal=True,  use_terminal_reward=True,  use_fatigue_dynamics=False)),
        ("Sparse only",     dict(use_r_goal=False, use_terminal_reward=True,  use_fatigue_dynamics=True)),
    ]

    table = {}
    for name, env_kwargs in ablations:
        print(f"\n  -- {name} : {env_kwargs} --")
        ppo = PPOAgent(lr=5e-3, gamma=0.95, n_episodes_per_update=4)
        for ep in range(1, N_TRAIN_EPISODES + 1):
            env = make_env(make_profile(ep), **env_kwargs)
            run_ppo(env, ppo, train=True)
            if ep % 200 == 0:
                print(f"    ep {ep:4d}")
        m = evaluate(name, ppo, run_ppo, N_EVAL_EPISODES, env_kwargs=env_kwargs)
        table[name] = m

    print("\n  Ablation summary:")
    print(f"  {'Variant':14s} {'Reward':>14s} {'Both':>6s} {'Budget%':>8s} {'Cal%':>6s}")
    for name, m in table.items():
        print(f"  {name:14s} {m['mean_reward']:+8.1f}±{m['ci95_reward']:>4.1f} "
              f"{m['constraint_ok_pct']:>5.0f}% {m['budget_pct']:>7.0f}% {m['calorie_pct']:>5.0f}%")

    out_path = os.path.join(os.path.dirname(__file__), "tasteflow_ablation.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"ablation_results": table}, f)
    print(f"\n  Saved → {out_path}")
    return table


# ── Entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true",
                        help="Additionally run the ablation study")
    parser.add_argument("--only-ablation", action="store_true",
                        help="Skip main training; only run the ablation study")
    args = parser.parse_args()

    if not args.only_ablation:
        train_and_evaluate()
    if args.ablation or args.only_ablation:
        ablation_study()
