# TasteFlow — RL Demo App (PPO + PPO-Lagrangian)

## Quick Start

```bash
pip install -r requirements.txt
python train.py              # trains PPO + PPO-L + pre-trains all baselines (~60s)
python train.py --ablation   # additionally runs the env-component ablation (~3 min)
streamlit run app.py         # opens in browser at localhost:8501
```

> **Note on retraining.** If you change `env.py` (state, reward, ablation flags) you
> must re-run `python train.py` to refresh `tasteflow_results.pkl` — the Streamlit
> demo loads cached weights and curves from that file.

## Troubleshooting

If `streamlit run app.py` fails with `streamlit is not recognized` or
`streamlit: command not found`, Streamlit is either not installed or is installed
in a different Python environment.

Run Streamlit through the active Python interpreter instead:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, you can also use the Python launcher:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If you prefer an isolated environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks virtual environment activation, enable scripts for the
current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Files

| File | Purpose |
|------|---------|
| `env.py` | TasteFlow MDP — 24-dim state, fatigue decay, budget/calorie tracking, weekly settlement |
| `agents.py` | 7 agents: Random, Greedy, ε-Greedy, Rule-Based, LinUCB, PPO, **PPO-Lagrangian** |
| `train.py` | Trains PPO + PPO-L, evaluates all agents, saves `tasteflow_results.pkl` |
| `app.py` | Streamlit demo — two modes, five tabs |
| `requirements.txt` | Python dependencies for local setup and Streamlit Community Cloud |
| `runtime.txt` | Python runtime version for Streamlit Community Cloud |

## Deployment

This app is built with Streamlit, so deploy it as a Python web app rather than GitHub Pages.
GitHub Pages only serves static HTML/CSS/JS and cannot run `app.py`.

Recommended deployment target: **Streamlit Community Cloud**.

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the GitHub repo.
3. Set the main file path to `app.py`.
4. Keep `requirements.txt`, `runtime.txt`, and `tasteflow_results.pkl` in the repo.
5. After each `git push` to the deployed branch, Streamlit Cloud will redeploy the app.

`tasteflow_results.pkl` is intentionally kept in the repo so the deployed demo can load trained
curves and agent weights immediately. If you retrain locally with `python train.py`, commit the
updated file when you want the hosted demo to show the new results.

## Demo Modes

### 📊 Watch the Agent Learn — 5 tabs

| Tab | Content |
|-----|---------|
| 📈 Learning Curve | PPO vs PPO-L reward curves + Lagrange multiplier evolution |
| 🏆 Agent Comparison | Reward / budget attainment / churn across all 7 agents |
| 🔄 Before vs After | Random agent vs trained PPO episode log |
| 🧠 State & Reward Breakdown | Fatigue heatmap, reward decomposition, goal tracking |
| **⚖️ PPO-L: Constraint Analysis** | **Lagrangian formulation, λ curves, constraint violation comparison, PPO vs PPO-L episode diff** |

### 🎮 Be the User
Interactive episode — choose PPO or PPO-L as your recommender, click Accept/Reject/Churn,
watch the state update and reward accumulate in real time.

## PPO-Lagrangian (PPO-L) — Primal-Dual Constrained RL

> **Naming note.** This is **PPO-Lagrangian** (Ray et al., 2019, *Benchmarking
> Safe Exploration in Deep RL*) — the standard primal-dual safe-RL baseline.
> It is **not** true CPO (Achiam et al., 2017), which uses trust-region constraints
> with conjugate gradient and second-order updates. Earlier drafts of this project
> called the agent "CPO"; that label was inaccurate and has been corrected.

PPO folds budget/calorie goals into the reward as *soft bonuses* (r_goal).
The agent can trade constraint satisfaction for acceptance reward.

PPO-L separates the primary objective from constraint enforcement
using the **Primal-Dual (Lagrangian) method**:

```
maximise  E[Σ r_accept_t]          ← pure acceptance, no r_goal
subject to  E[budget_fraction]   ≤ 0.95
            E[calorie_fraction]  ≤ 0.95
```

The multipliers λ_budget and λ_calorie are learned alongside the policy:
- Constraint violated → λ grows → expensive/caloric meals penalised harder
- Constraint satisfied → λ shrinks → policy relaxes to optimise acceptance

At convergence the Lagrangian saddle-point satisfies KKT complementary
slackness: λ* · (C − d) = 0.

## Results (800 training episodes, 50 eval episodes, fair comparison)

All non-trivial agents — including Greedy, ε-Greedy, and LinUCB — are pre-trained
for the same 800 episodes before evaluation, so the comparison is apples-to-apples.
Confidence intervals are 95% (normal approx, n=50).

| Agent | Reward (95% CI) | Budget Met | Calories Met | Both Goals | Churn |
|-------|------------------|-----------|-------------|-----------|-------|
| Random        | +133.4 ± 4.0  |  24% | 100% |  24% | 0% |
| Greedy (ε=0)  | +135.6 ± 0.3  |   0% | 100% |   0% | 0% |
| ε-Greedy (ε=0.1) | +182.2 ± 2.5 | 100% |  94% |  94% | 0% |
| Rule-Based    | +126.1 ± 4.4  |  34% | 100% |  34% | 0% |
| LinUCB        | +133.4 ± 0.5  |   0% | 100% |   0% | 0% |
| **PPO**       | **+186.5 ± 1.6** | **98%** | **100%** | **98%** | 0% |
| **PPO-L**     | **+183.0 ± 2.1** |  96% | 100% |  96% | 0% |

**Take-aways**

- **PPO** wins on reward via explicit goal shaping (`r_goal`) folded into the per-step reward.
- **PPO-L** matches PPO on both reward and constraint satisfaction *without* using `r_goal` — it
  enforces the budget purely through the Lagrange multiplier λ_b, which converges around 4.6.
- **ε-Greedy** is a surprisingly strong baseline once given the same training budget, lifting
  Greedy from 0% → 100% budget through 10% exploration alone.
- **LinUCB / Greedy** never satisfy the budget constraint: they have no notion of episode-level
  pacing — each pick maximises immediate reward.

## Ablation Study (PPO under env-component flips, 800 train eps)

Run with `python train.py --ablation`. PPO is retrained from scratch for each variant.

| Variant       | Reward | Both Goals | Budget% | Note |
|---------------|--------|------------|---------|------|
| Full          | +182.2 ± 3.0 | 92% | 92% | Baseline (all components on) |
| No r_goal     | +146.1 ± 2.4 | 94% | 94% | Reward drops; constraint still met (terminal bonus suffices) |
| No terminal   | +133.2 ± 1.6 | 82% | 82% | Both reward and constraint suffer without sparse end-reward |
| No fatigue    | +186.7 ± 2.1 | 96% | 96% | Slightly *better* — fatigue is the main source of reward variance |
| Sparse only   | +146.1 ± 2.4 | 94% | 94% | r_goal off, terminal on — equivalent to "No r_goal" |

**Insights**

- The **terminal bonus** is the most important single component. Removing it costs 49 reward
  points and 10pp of constraint satisfaction — credit assignment relies on that sparse signal
  to align per-step decisions with weekly goals.
- `r_goal` (per-step shaping) primarily boosts *reward magnitude*, not constraint satisfaction.
- Disabling **fatigue dynamics** removes a noise source from the user model and the agent
  performs marginally better; this validates that the fatigue penalty is real and meaningful.

## Business Case

**Product.** A weekly meal-recommendation companion for office workers / students who
want to stay within a budget and a calorie target without spending mental energy planning
each meal. The agent observes the user's weekly state (budget remaining, calories eaten,
recent picks, time of day) and proposes one meal per slot.

**Why RL, not a recipe app or rules engine?**

| Need | Why a static rule fails | What RL provides |
|------|------------------------|------------------|
| Weekly pacing of budget/calories | Static rules can't trade off "save now, splurge later" | Sequential MDP optimises return over the whole horizon |
| Personalisation that adapts | A rule needs to be re-coded per user | Policy is a function of state — adapts via online preference EMA |
| Non-stationary preferences | Rules need explicit override paths | Policy gradient adapts to drift (preferred-cat shift at step 10) |
| Safety / nutrition constraints | Hard-coded constraints have no reward trade-off | PPO-Lagrangian enforces via Lagrangian, learns dual price of each constraint |

**Monetisation paths.** (i) B2C subscription with personalised weekly plans;
(ii) B2B partnership with food-delivery platforms — surface high-margin items
that satisfy the user's dietary constraints; (iii) corporate wellness — employers
buy seats for staff; (iv) restaurant recommendation marketplace where vendors bid
for action-mask inclusion within compliant meals.

**Evaluation against this case.** The simulator + RL pipeline shown above is a
pre-product validation of the **mechanism**: PPO and PPO-L can learn to satisfy weekly
budget/calorie constraints from interaction signals alone, and outperform both
heuristic and bandit baselines that lack a notion of episode-level state. A real
deployment would replace `UserProfile` with logged user data and a learned reward
model, and replace the 14-meal pool with a live restaurant catalogue.

## Implementation Notes

- **PPO update** uses the proper clipped-surrogate gradient (`grad ∝ ratio·adv·(π − one_hot_a)`
  when the unclipped term binds, zero in the strict-clip zone) — *not* the
  REINFORCE-style approximation that earlier versions used. Entropy gradient is also
  back-propagated, so the entropy bonus actually influences updates.
- **PPO-Lagrangian** uses **two-timescale** primal-dual: λ_b, λ_c update *every*
  episode (fast), while the policy updates every `n_episodes_per_update` episodes
  (slow). PPO-L returns are *not* z-normalised — that destroys the absolute scale
  of the Lagrangian penalty `−λ·cost` relative to `r_accept`.
- **State** is 24-dim including a real `time_since_last` (hours since last accepted meal,
  capped at 24). Earlier versions hardcoded this to 4.0; that bug is fixed.
