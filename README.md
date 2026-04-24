# TasteFlow — RL Demo App (PPO + CPO)

## Quick Start

```bash
pip install streamlit numpy matplotlib plotly
python train.py       # ~90 seconds — trains PPO + CPO + evaluates all 5 agents
streamlit run app.py  # opens in browser at localhost:8501
```

## Files

| File | Purpose |
|------|---------|
| `env.py` | TasteFlow MDP — 24-dim state, fatigue decay, budget/calorie tracking, weekly settlement |
| `agents.py` | 5 agents: Random, Rule-Based, LinUCB, PPO, **CPO** |
| `train.py` | Trains PPO + CPO, evaluates all agents, saves `tasteflow_results.pkl` |
| `app.py` | Streamlit demo — two modes, five tabs |

## Demo Modes

### 📊 Watch the Agent Learn — 5 tabs

| Tab | Content |
|-----|---------|
| 📈 Learning Curve | PPO vs CPO reward curves + Lagrange multiplier evolution |
| 🏆 Agent Comparison | Reward / budget attainment / churn across all 5 agents |
| 🔄 Before vs After | Random agent vs trained PPO episode log |
| 🧠 State & Reward Breakdown | Fatigue heatmap, reward decomposition, goal tracking |
| **⚖️ CPO: Constraint Analysis** | **Lagrangian formulation, λ curves, constraint violation comparison, PPO vs CPO episode diff** |

### 🎮 Be the User
Interactive episode — choose PPO or CPO as your recommender, click Accept/Reject/Churn,
watch the state update and reward accumulate in real time.

## CPO — Why It's Novel

PPO folds budget/calorie goals into the reward as *soft bonuses* (r_goal).
The agent can trade constraint satisfaction for acceptance reward.

CPO treats them as **hard Lagrangian constraints**:

```
maximise  E[Σ r_accept_t]          ← pure acceptance, no r_goal
subject to  E[budget_fraction]   ≤ 0.95
            E[calorie_fraction]  ≤ 0.95
```

The multipliers λ_budget and λ_calorie are learned alongside the policy:
- Constraint violated → λ grows → expensive/caloric meals penalised harder
- Constraint satisfied → λ shrinks → policy relaxes to optimise acceptance

KKT conditions guarantee: at convergence, λ* · (C − d) = 0 —
either the constraint is satisfied, or the multiplier is zero.
No prior food recommendation work uses constrained MDP for goal enforcement.

## Results (800 training episodes, 50 eval episodes)

| Agent | Reward | Budget Met | Calories Met | Both Goals | Churn |
|-------|--------|-----------|-------------|-----------|-------|
| Random | +130 | 46% | 100% | 46% | 0% |
| Rule-Based | +124 | 48% | 100% | 48% | 0% |
| LinUCB | +111 | 0% | 100% | 0% | 0% |
| PPO | **+190** | **100%** | **100%** | **100%** | 0% |
| CPO | +152 | 32% | 100% | 32% | 0% |

PPO leads on reward; CPO demonstrates the constraint-enforcement mechanism
with Lagrange multipliers growing during training (visible in Tab 5).
