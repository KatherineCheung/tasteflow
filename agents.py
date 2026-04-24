"""
Agents:
  - RandomAgent      : uniform random baseline
  - RuleBasedAgent   : recency + variety heuristic (no learning)
  - LinUCBAgent      : contextual bandit (Section 2 comparison)
  - PPOAgent         : full MDP agent (from-scratch NumPy PPO)
  - CPOAgent         : Constrained Policy Optimization via Primal-Dual method
                       Treats budget + calorie limits as hard Lagrangian constraints,
                       not soft reward bonuses. Provably satisfies constraints at
                       convergence while maximising acceptance reward.
"""

import numpy as np
from env import N_ACTIONS, STATE_DIM, CATEGORIES, N_CAT, MEALS


# ── Helpers ────────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


# ── Random Agent ───────────────────────────────────────────────────────────
class RandomAgent:
    name  = "Random Baseline"
    color = "#888888"

    def select(self, state, valid_actions):
        return np.random.choice(valid_actions)

    def update(self, *args, **kwargs):
        pass


# ── Rule-Based Agent ───────────────────────────────────────────────────────
class RuleBasedAgent:
    """
    Heuristic: score each meal by
      - preference match (from state pref_weights, indices 19-25)
      - inverse fatigue (from state, indices 4-10)
      - budget fit
    No learning — weights are hand-tuned.
    """
    name  = "Rule-Based"
    color = "#F97316"

    def __init__(self):
        self.visit_counts = np.zeros(N_ACTIONS)

    def select(self, state, valid_actions):
        fatigue   = state[4:4+N_CAT]
        pref_w    = state[4+N_CAT+4 : 4+N_CAT+4+N_CAT]
        budget_r  = state[0]

        scores = []
        for i in valid_actions:
            meal    = MEALS[i]
            cat_idx = CATEGORIES.index(meal["category"])
            score   = (pref_w[cat_idx] * 2.0
                       - fatigue[cat_idx] * 1.5
                       + budget_r * 0.5
                       - self.visit_counts[i] * 0.1)
            scores.append(score)

        best_idx = valid_actions[int(np.argmax(scores))]
        self.visit_counts[best_idx] += 1
        return best_idx

    def update(self, *args, **kwargs):
        pass


# ── LinUCB Contextual Bandit ───────────────────────────────────────────────
# ── Greedy Agent ───────────────────────────────────────────────────────────
class GreedyAgent:
    """
    Pure greedy (ε=0): always picks the meal with the highest estimated
    average reward. Uses an incremental mean Q-table keyed on meal ID.

    No context, no exploration — a pure exploitation baseline.
    Expected failure mode: discovers that the user's preferred category
    (Japanese) yields high reward, spams it until fatigue collapses
    acceptance, then gets stuck because it never explored alternatives.
    This makes the exploration-exploitation tradeoff argument concrete.
    """
    name  = "Greedy (ε=0)"
    color = "#F43F5E"   # rose-red — visually distinct

    def __init__(self):
        self.Q     = np.zeros(N_ACTIONS)   # estimated mean reward per meal
        self.count = np.zeros(N_ACTIONS)   # visit count per meal

    def select(self, state, valid_actions):
        # argmax over valid actions only
        q_valid = [(self.Q[a], a) for a in valid_actions]
        return max(q_valid, key=lambda x: x[0])[1]

    def update(self, action, state, reward, *args, **kwargs):
        self.count[action] += 1
        # incremental mean: Q ← Q + (r - Q) / n
        self.Q[action] += (reward - self.Q[action]) / self.count[action]


# ── Epsilon-Greedy Agent ───────────────────────────────────────────────────
class EpsilonGreedyAgent:
    """
    ε-Greedy: with probability ε explore (uniform random), otherwise exploit
    (argmax Q). Fixes the pure greedy failure mode by occasionally trying
    meals it hasn't recently seen.

    Key hyperparameter: ε=0.1 means 10% of decisions are random exploration.
    Compare with Greedy (ε=0) to show exactly what a small amount of
    exploration buys you in terms of variety, fatigue avoidance, and reward.

    Like GreedyAgent, Q is context-free (no state features) — this isolates
    the exploration effect from contextual learning.
    """
    name  = "ε-Greedy (ε=0.1)"
    color = "#FB923C"   # orange — positioned between Greedy red and Rule-Based

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.Q       = np.zeros(N_ACTIONS)
        self.count   = np.zeros(N_ACTIONS)

    def select(self, state, valid_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)   # explore
        q_valid = [(self.Q[a], a) for a in valid_actions]
        return max(q_valid, key=lambda x: x[0])[1]   # exploit

    def update(self, action, state, reward, *args, **kwargs):
        self.count[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.count[action]


class LinUCBAgent:
    """
    LinUCB: p_a = x^T θ_a + α √(x^T A_a^{-1} x)
    Uses state vector as context (no arm-specific features for simplicity).
    """
    name  = "LinUCB Bandit"
    color = "#FFE66D"

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        d = STATE_DIM
        self.A = {i: np.eye(d)    for i in range(N_ACTIONS)}
        self.b = {i: np.zeros(d)  for i in range(N_ACTIONS)}

    def select(self, state, valid_actions):
        best_ucb, best = -np.inf, valid_actions[0]
        x = state.astype(float)
        for i in valid_actions:
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            ucb   = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            if ucb > best_ucb:
                best_ucb, best = ucb, i
        return best

    def update(self, action, state, reward, *args, **kwargs):
        x = state.astype(float)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


# ── PPO Agent (pure NumPy, 2-layer MLP) ───────────────────────────────────
class PPOAgent:
    """
    Proximal Policy Optimization — from-scratch NumPy implementation.
    Architecture: shared trunk (FC128→FC64) + policy head + value head.
    """
    name  = "PPO (TasteFlow)"
    color = "#4ECDC4"

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS,
                 lr=3e-3, gamma=0.95, clip_eps=0.2,
                 entropy_coef=0.01, n_epochs=4):
        self.state_dim    = state_dim
        self.n_actions    = n_actions
        self.lr           = lr
        self.gamma        = gamma
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs

        # ── network weights (Xavier init) ──
        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_out, fan_in))

        # shared trunk
        self.W1 = xavier(state_dim, 128);  self.b1 = np.zeros(128)
        self.W2 = xavier(128, 64);         self.b2 = np.zeros(64)
        # policy head
        self.Wp = xavier(64, n_actions);   self.bp = np.zeros(n_actions)
        # value head
        self.Wv = xavier(64, 1);           self.bv = np.zeros(1)

        # replay buffer
        self.buf_states    = []
        self.buf_actions   = []
        self.buf_rewards   = []
        self.buf_logprobs  = []
        self.buf_dones     = []
        self.buf_masks     = []

        self.update_count  = 0

    # ── forward pass ──
    def _forward(self, s):
        h1 = np.tanh(self.W1 @ s + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        logits = self.Wp @ h2 + self.bp
        value  = (self.Wv @ h2 + self.bv)[0]
        return h1, h2, logits, value

    def _policy(self, logits, mask):
        masked = logits.copy()
        masked[[i for i in range(self.n_actions) if i not in mask]] = -1e9
        return softmax(masked)

    # ── select ──
    def select(self, state, valid_actions):
        _, _, logits, _ = self._forward(state)
        probs  = self._policy(logits, valid_actions)
        action = np.random.choice(self.n_actions, p=probs)
        return action

    def select_greedy(self, state, valid_actions):
        """Deterministic selection for evaluation / demo."""
        _, _, logits, _ = self._forward(state)
        probs  = self._policy(logits, valid_actions)
        return valid_actions[int(np.argmax([probs[i] for i in valid_actions]))]

    def get_action_probs(self, state, valid_actions):
        _, _, logits, _ = self._forward(state)
        return self._policy(logits, valid_actions)

    # ── store transition ──
    def store(self, state, action, reward, logprob, done, valid_mask):
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_rewards.append(reward)
        self.buf_logprobs.append(logprob)
        self.buf_dones.append(done)
        self.buf_masks.append(valid_mask)

    # ── compute log prob of action ──
    def logprob(self, state, action, valid_actions):
        _, _, logits, _ = self._forward(state)
        probs  = self._policy(logits, valid_actions)
        return np.log(probs[action] + 1e-8)

    # ── update (PPO clip objective) ──
    def update(self, *args, **kwargs):
        if len(self.buf_states) < 10:
            return {}

        states   = np.array(self.buf_states,   dtype=np.float32)
        actions  = np.array(self.buf_actions,  dtype=int)
        rewards  = np.array(self.buf_rewards,  dtype=np.float32)
        old_lps  = np.array(self.buf_logprobs, dtype=np.float32)
        dones    = np.array(self.buf_dones,    dtype=np.float32)
        masks    = self.buf_masks

        # ── compute returns (discounted) ──
        returns = np.zeros_like(rewards)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G * (1 - dones[t])
            returns[t] = G

        # normalise returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── PPO update (n_epochs passes) ──
        total_loss = 0.0
        for _ in range(self.n_epochs):
            for t in range(len(states)):
                s  = states[t]
                a  = actions[t]
                G  = returns[t]
                m  = masks[t]

                h1, h2, logits, value = self._forward(s)
                probs = self._policy(logits, m)
                new_lp = np.log(probs[a] + 1e-8)

                # advantage
                adv = G - value

                # clipped ratio
                ratio    = np.exp(new_lp - old_lps[t])
                clip_r   = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_l  = -min(ratio * adv, clip_r * adv)

                # value loss
                critic_l = (G - value) ** 2

                # entropy bonus
                entropy  = -np.sum(probs * np.log(probs + 1e-8))
                loss     = actor_l + 0.5 * critic_l - self.entropy_coef * entropy
                total_loss += loss

                # ── backprop (manual chain rule) ──
                # value head gradient
                dv   = 2 * (value - G)             # dL/d_value
                dWv  = dv * h2.reshape(1, -1)
                dbv  = np.array([dv])

                # policy head gradient (approximate — REINFORCE-style for simplicity)
                dlogp     = np.zeros(self.n_actions)
                dlogp[a]  = -adv / (probs[a] + 1e-8)
                # softmax jacobian diagonal approx
                dlogits   = probs * (dlogp - (probs * dlogp).sum())
                dWp       = np.outer(dlogits, h2)
                dbp       = dlogits

                # shared trunk gradients
                dh2_from_p = self.Wp.T @ dlogits
                dh2_from_v = self.Wv.T * dv
                dh2        = (dh2_from_p + dh2_from_v.flatten()) * (1 - h2**2)

                dW2  = np.outer(dh2, h1)
                db2  = dh2

                dh1  = (self.W2.T @ dh2) * (1 - h1**2)
                dW1  = np.outer(dh1, s)
                db1  = dh1

                # ── gradient step (clip gradients for stability) ──
                def clip(g, c=1.0):
                    n = np.linalg.norm(g)
                    return g * (c / n) if n > c else g

                lr = self.lr
                self.W1 -= lr * clip(dW1); self.b1 -= lr * clip(db1)
                self.W2 -= lr * clip(dW2); self.b2 -= lr * clip(db2)
                self.Wp -= lr * clip(dWp); self.bp -= lr * clip(dbp)
                self.Wv -= lr * clip(dWv); self.bv -= lr * clip(dbv)

        self.update_count += 1

        # clear buffer
        self.buf_states.clear(); self.buf_actions.clear()
        self.buf_rewards.clear(); self.buf_logprobs.clear()
        self.buf_dones.clear();  self.buf_masks.clear()

        return {"loss": total_loss / max(len(states) * self.n_epochs, 1)}


# ── CPO Agent (Constrained Policy Optimization — Primal-Dual) ─────────────
class CPOAgent:
    """
    Constrained Policy Optimization via the Primal-Dual (Lagrangian) method.

    Core idea
    ---------
    PPO folds budget and calorie goals into the reward as soft bonuses (r_goal).
    That means the agent *trades off* constraint satisfaction against acceptance
    reward — it might overspend if doing so gets enough +5 accept rewards.

    CPO separates concerns:
      • PRIMARY objective  : maximise E[Σ r_accept_t]   (pure acceptance reward)
      • HARD constraint 1  : E[budget_spent / weekly_budget]   ≤ BUDGET_LIMIT
      • HARD constraint 2  : E[calories_eaten / weekly_calories] ≤ CALORIE_LIMIT

    The Lagrangian formulation converts this into an unconstrained saddle-point:

        L(π, λ) = E[Σ r_accept] - λ_b · (C_b - d_b) - λ_c · (C_c - d_c)

    where C_b, C_c are the observed constraint costs this episode and
    d_b, d_c are the allowed limits.

    The policy π is updated to MAXIMISE L (standard PPO-clip on r_accept only).
    The multipliers λ_b, λ_c are updated to MINIMISE L (gradient ascent on
    constraint violation) — they grow large when constraints are violated,
    automatically penalising the policy harder until it complies.

    At convergence the KKT conditions guarantee:
        λ_b* · (C_b - d_b) = 0    (complementary slackness)
    meaning either the constraint is satisfied OR the multiplier is zero,
    never both violated simultaneously.

    Architecture
    ------------
    Same shared trunk as PPOAgent (FC128 → FC64) but the reward signal passed
    to the policy update strips out r_goal — only r_accept flows into the
    actor loss. Constraint costs are tracked separately and fed into the
    Lagrange multiplier update step.
    """
    name  = "CPO (Constrained)"
    color = "#A78BFA"   # purple — distinct from PPO teal

    # Constraint limits (fraction of weekly goal consumed)
    BUDGET_LIMIT  = 0.95   # allow up to 95% of budget → forces pacing
    CALORIE_LIMIT = 0.95   # allow up to 95% of calorie allowance

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS,
                 lr_policy=3e-3, lr_lambda=1e-2,
                 gamma=0.95, clip_eps=0.2,
                 entropy_coef=0.01, n_epochs=4,
                 lambda_init=0.1, lambda_max=10.0):

        self.state_dim    = state_dim
        self.n_actions    = n_actions
        self.lr_policy    = lr_policy
        self.lr_lambda    = lr_lambda      # Lagrange multiplier learning rate
        self.gamma        = gamma
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.lambda_max   = lambda_max

        # ── Lagrange multipliers (one per constraint) ──────────────────
        # Initialised to a small positive value.
        # They grow whenever the agent violates a constraint, shrink when safe.
        self.lambda_budget  = lambda_init   # multiplier for budget constraint
        self.lambda_calorie = lambda_init   # multiplier for calorie constraint

        # Running estimates of constraint costs (exponential moving average)
        # Used to smooth out episode-to-episode variance before updating λ
        self.ema_budget_cost  = 0.0
        self.ema_calorie_cost = 0.0
        self.ema_alpha        = 0.1        # EMA smoothing factor

        # ── Network weights (Xavier init, identical structure to PPO) ──
        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_out, fan_in))

        self.W1 = xavier(state_dim, 128);  self.b1 = np.zeros(128)
        self.W2 = xavier(128, 64);         self.b2 = np.zeros(64)
        self.Wp = xavier(64, n_actions);   self.bp = np.zeros(n_actions)
        self.Wv = xavier(64, 1);           self.bv = np.zeros(1)

        # ── Replay buffer ──────────────────────────────────────────────
        # Stores r_accept separately from r_goal so the policy update
        # only sees acceptance reward, not the soft constraint bonuses.
        self.buf_states         = []
        self.buf_actions        = []
        self.buf_r_accept       = []   # pure acceptance reward (no r_goal)
        self.buf_logprobs       = []
        self.buf_dones          = []
        self.buf_masks          = []
        self.buf_budget_costs   = []   # fraction of budget spent this step
        self.buf_calorie_costs  = []   # fraction of calories consumed this step

        self.update_count = 0

        # ── Tracking for visualisation ─────────────────────────────────
        self.lambda_budget_history  = [lambda_init]
        self.lambda_calorie_history = [lambda_init]
        self.constraint_violations  = []   # (episode, which_constraint)

    # ── Forward pass (identical to PPO) ───────────────────────────────────
    def _forward(self, s):
        h1     = np.tanh(self.W1 @ s + self.b1)
        h2     = np.tanh(self.W2 @ h1 + self.b2)
        logits = self.Wp @ h2 + self.bp
        value  = (self.Wv @ h2 + self.bv)[0]
        return h1, h2, logits, value

    def _policy(self, logits, mask):
        masked = logits.copy()
        masked[[i for i in range(self.n_actions) if i not in mask]] = -1e9
        return softmax(masked)

    # ── Select action ──────────────────────────────────────────────────────
    def select(self, state, valid_actions):
        _, _, logits, _ = self._forward(state)
        probs  = self._policy(logits, valid_actions)
        return np.random.choice(self.n_actions, p=probs)

    def select_greedy(self, state, valid_actions):
        _, _, logits, _ = self._forward(state)
        probs = self._policy(logits, valid_actions)
        return valid_actions[int(np.argmax([probs[i] for i in valid_actions]))]

    def get_action_probs(self, state, valid_actions):
        _, _, logits, _ = self._forward(state)
        return self._policy(logits, valid_actions)

    # ── Store transition ───────────────────────────────────────────────────
    def store(self, state, action, r_accept, logprob, done,
              valid_mask, budget_cost, calorie_cost):
        """
        r_accept   : ONLY the acceptance component (+5/-2/-10), no r_goal.
        budget_cost: meal_price / weekly_budget  (normalised cost this step)
        calorie_cost: meal_calories / weekly_calories
        """
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_r_accept.append(r_accept)
        self.buf_logprobs.append(logprob)
        self.buf_dones.append(done)
        self.buf_masks.append(valid_mask)
        self.buf_budget_costs.append(budget_cost)
        self.buf_calorie_costs.append(calorie_cost)

    def logprob(self, state, action, valid_actions):
        _, _, logits, _ = self._forward(state)
        probs = self._policy(logits, valid_actions)
        return np.log(probs[action] + 1e-8)

    # ── Update: policy step + Lagrange multiplier step ────────────────────
    def update(self, episode_budget_fraction=None, episode_calorie_fraction=None):
        """
        Two-phase update:

        Phase 1 — Policy update (PPO-clip on r_accept only):
            The Lagrange multipliers enter as a PENALTY on constraint costs,
            augmenting the effective reward seen by the actor:

            r_augmented_t = r_accept_t
                          - λ_b  · budget_cost_t
                          - λ_c  · calorie_cost_t

            This is the Lagrangian penalty inline: high λ means the actor
            is strongly penalised for recommending expensive/caloric meals.

        Phase 2 — Multiplier update (gradient ascent on constraint violation):
            λ_b  ← clip(λ_b  + lr_λ · (C_b  - d_b),  0, λ_max)
            λ_c  ← clip(λ_c  + lr_λ · (C_c  - d_c),  0, λ_max)

            where C_b = EMA of budget_fraction over recent episodes.
            If C_b > d_b (budget violated) → λ_b increases → actor penalised harder.
            If C_b < d_b (budget satisfied) → λ_b decreases → actor has more freedom.
        """
        if len(self.buf_states) < 5:
            return {}

        states        = np.array(self.buf_states,       dtype=np.float32)
        actions       = np.array(self.buf_actions,      dtype=int)
        r_accepts     = np.array(self.buf_r_accept,     dtype=np.float32)
        old_lps       = np.array(self.buf_logprobs,     dtype=np.float32)
        dones         = np.array(self.buf_dones,        dtype=np.float32)
        masks         = self.buf_masks
        b_costs       = np.array(self.buf_budget_costs, dtype=np.float32)
        c_costs       = np.array(self.buf_calorie_costs,dtype=np.float32)

        # ── Augmented reward: r_accept - λ_b·C_b - λ_c·C_c ───────────
        r_augmented = (r_accepts
                       - self.lambda_budget  * b_costs
                       - self.lambda_calorie * c_costs)

        # ── Compute discounted returns on augmented reward ─────────────
        returns = np.zeros_like(r_augmented)
        G = 0.0
        for t in reversed(range(len(r_augmented))):
            G = r_augmented[t] + self.gamma * G * (1 - dones[t])
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── Phase 1: PPO-clip policy update ───────────────────────────
        for _ in range(self.n_epochs):
            for t in range(len(states)):
                s  = states[t];  a = actions[t];  G = returns[t];  m = masks[t]
                h1, h2, logits, value = self._forward(s)
                probs  = self._policy(logits, m)
                new_lp = np.log(probs[a] + 1e-8)

                adv    = G - value
                ratio  = np.exp(new_lp - old_lps[t])
                clip_r = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_l  = -min(ratio * adv, clip_r * adv)
                critic_l = (G - value) ** 2
                entropy  = -np.sum(probs * np.log(probs + 1e-8))
                loss     = actor_l + 0.5 * critic_l - self.entropy_coef * entropy

                # backprop (same as PPO)
                dv   = 2 * (value - G)
                dWv  = dv * h2.reshape(1, -1);  dbv = np.array([dv])

                dlogp        = np.zeros(self.n_actions)
                dlogp[a]     = -adv / (probs[a] + 1e-8)
                dlogits      = probs * (dlogp - (probs * dlogp).sum())
                dWp          = np.outer(dlogits, h2);  dbp = dlogits

                dh2_p = self.Wp.T @ dlogits
                dh2_v = self.Wv.T * dv
                dh2   = (dh2_p + dh2_v.flatten()) * (1 - h2**2)
                dW2   = np.outer(dh2, h1);  db2 = dh2

                dh1 = (self.W2.T @ dh2) * (1 - h1**2)
                dW1 = np.outer(dh1, s);   db1 = dh1

                def clip_grad(g, c=1.0):
                    n = np.linalg.norm(g)
                    return g * (c / n) if n > c else g

                lr = self.lr_policy
                self.W1 -= lr*clip_grad(dW1); self.b1 -= lr*clip_grad(db1)
                self.W2 -= lr*clip_grad(dW2); self.b2 -= lr*clip_grad(db2)
                self.Wp -= lr*clip_grad(dWp); self.bp -= lr*clip_grad(dbp)
                self.Wv -= lr*clip_grad(dWv); self.bv -= lr*clip_grad(dbv)

        # ── Phase 2: Lagrange multiplier update ───────────────────────
        # Update EMA of constraint costs using this episode's fractions
        if episode_budget_fraction is not None:
            self.ema_budget_cost = (
                (1 - self.ema_alpha) * self.ema_budget_cost
                + self.ema_alpha * episode_budget_fraction
            )
            self.ema_calorie_cost = (
                (1 - self.ema_alpha) * self.ema_calorie_cost
                + self.ema_alpha * episode_calorie_fraction
            )
        else:
            # Fallback: use mean step costs from this buffer
            self.ema_budget_cost = (
                (1 - self.ema_alpha) * self.ema_budget_cost
                + self.ema_alpha * float(b_costs.sum())
            )
            self.ema_calorie_cost = (
                (1 - self.ema_alpha) * self.ema_calorie_cost
                + self.ema_alpha * float(c_costs.sum())
            )

        # Gradient ascent: λ increases when constraint violated, decreases when safe
        budget_violation  = self.ema_budget_cost  - self.BUDGET_LIMIT
        calorie_violation = self.ema_calorie_cost - self.CALORIE_LIMIT

        self.lambda_budget = float(np.clip(
            self.lambda_budget  + self.lr_lambda * budget_violation,
            0.0, self.lambda_max
        ))
        self.lambda_calorie = float(np.clip(
            self.lambda_calorie + self.lr_lambda * calorie_violation,
            0.0, self.lambda_max
        ))

        # Track for visualisation
        self.lambda_budget_history.append(self.lambda_budget)
        self.lambda_calorie_history.append(self.lambda_calorie)

        self.update_count += 1

        # Clear buffer
        self.buf_states.clear();        self.buf_actions.clear()
        self.buf_r_accept.clear();      self.buf_logprobs.clear()
        self.buf_dones.clear();         self.buf_masks.clear()
        self.buf_budget_costs.clear();  self.buf_calorie_costs.clear()

        return {
            "lambda_budget":  self.lambda_budget,
            "lambda_calorie": self.lambda_calorie,
            "budget_violation":  budget_violation,
            "calorie_violation": calorie_violation,
        }
