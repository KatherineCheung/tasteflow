"""
TasteFlow MDP Environment
State: 24-dim vector (goal tracking, fatigue, temporal, preference)
Episode: 1 week = up to 21 meal steps (3/day × 7 days)
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
CATEGORIES   = ["Japanese", "Chinese", "Korean", "Fast Food", "Light", "Hot Pot", "Western"]
MEAL_TIMES   = ["Breakfast", "Lunch", "Dinner"]
DAYS         = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N_CAT        = len(CATEGORIES)
N_MEALS_DAY  = len(MEAL_TIMES)
N_DAYS       = 7
MAX_STEPS    = N_DAYS * N_MEALS_DAY   # 21

FATIGUE_DECAY   = 0.6      # λ — half-life of ~2.5 meals
CHURN_LIMIT     = 4        # consecutive churns → episode ends early
STATE_DIM       = 4 + N_CAT + 4 + (N_CAT + 2)  # 4+7+4+9 = 24

# Clock hours of each meal slot (used to compute real time_since_last)
MEAL_HOURS      = [8, 13, 19]   # Breakfast 08:00, Lunch 13:00, Dinner 19:00

CATEGORY_EMOJIS = {
    "Japanese":  "🍜", "Chinese": "🥡", "Korean":  "🍲",
    "Fast Food": "🍔", "Light":   "🥗", "Hot Pot": "🫕", "Western": "🥩",
}

MEAL_EMOJIS = {
    "Ramen": "🍜",
    "Sushi Set": "🍣",
    "Dim Sum": "🥟",
    "Noodle Soup": "🍲",
    "Korean BBQ": "🍖",
    "Bibimbap": "🍚",
    "Burger": "🍔",
    "Fried Chicken": "🍗",
    "Salad Bowl": "🥗",
    "Rice Bowl": "🍚",
    "Hot Pot": "🫕",
    "Mala Hot Pot": "🌶️",
    "Pasta": "🍝",
    "Steak Set": "🥩",
}

# ── Candidate meal pool ────────────────────────────────────────────────────
def make_meals():
    """Fixed pool of 14 candidate meals (2 per category)."""
    pool = []
    specs = [
        ("Ramen",          "Japanese",  800,  12, ["Lunch","Dinner"], ["comfort", "warm", "noodle"]),
        ("Sushi Set",      "Japanese",  600,  18, ["Lunch","Dinner"], ["fresh", "high-protein", "light"]),
        ("Dim Sum",        "Chinese",   700,  10, ["Breakfast","Lunch"], ["shareable", "comfort", "savory"]),
        ("Noodle Soup",    "Chinese",   650,   9, ["Breakfast","Lunch","Dinner"], ["warm", "noodle", "budget"]),
        ("Korean BBQ",     "Korean",    900,  20, ["Lunch","Dinner"], ["high-protein", "social", "grilled"]),
        ("Bibimbap",       "Korean",    750,  13, ["Lunch","Dinner"], ["balanced", "spicy", "rice"]),
        ("Burger",         "Fast Food", 900,   8, ["Breakfast","Lunch","Dinner"], ["quick", "comfort", "budget"]),
        ("Fried Chicken",  "Fast Food", 950,   7, ["Lunch","Dinner"], ["quick", "comfort", "crispy"]),
        ("Salad Bowl",     "Light",     400,  10, ["Breakfast","Lunch"], ["light", "healthy", "vegetarian"]),
        ("Rice Bowl",      "Light",     500,  11, ["Lunch","Dinner"], ["balanced", "healthy", "rice"]),
        ("Hot Pot",        "Hot Pot",  1100,  25, ["Dinner"], ["social", "warm", "premium"]),
        ("Mala Hot Pot",   "Hot Pot",  1050,  22, ["Dinner"], ["social", "warm", "spicy"]),
        ("Pasta",          "Western",   750,  14, ["Lunch","Dinner"], ["comfort", "quick", "savory"]),
        ("Steak Set",      "Western",   800,  28, ["Dinner"], ["high-protein", "premium", "grilled"]),
    ]
    for name, cat, cal, price, meal_types, tags in specs:
        pool.append({
            "name": name, "category": cat,
            "calories": cal, "price": price,
            "meal_types": meal_types,
            "tags": tags,
            "emoji": MEAL_EMOJIS.get(name, CATEGORY_EMOJIS[cat]),
        })
    return pool

MEALS = make_meals()
N_ACTIONS = len(MEALS)


# ── State builder ──────────────────────────────────────────────────────────
def build_state(budget_rem, cal_rem, days_rem, meals_rem,
                fatigue, meal_time_idx, day_idx,
                time_since_last, is_weekend,
                pref_weights, avg_price_pref, consec_churns):
    """Construct the 24-dim state vector."""
    group_a = np.array([
        budget_rem, cal_rem,
        days_rem / N_DAYS, meals_rem / MAX_STEPS
    ])
    group_b = np.array(fatigue)                          # 7-dim
    group_c = np.array([
        meal_time_idx / (N_MEALS_DAY - 1),
        day_idx / (N_DAYS - 1),
        min(time_since_last / 12.0, 1.0),
        float(is_weekend)
    ])
    group_d = np.array(
        list(pref_weights) +
        [avg_price_pref / 30.0, min(consec_churns / CHURN_LIMIT, 1.0)]
    )                                                    # 9-dim
    return np.concatenate([group_a, group_b, group_c, group_d]).astype(np.float32)


# ── Environment ────────────────────────────────────────────────────────────
class TasteFlowEnv:
    def __init__(self, weekly_budget=50.0, weekly_calories=10000,
                 user_profile=None,
                 use_r_goal=True, use_terminal_reward=True,
                 use_fatigue_dynamics=True):
        """
        Ablation flags (all True = full TasteFlow MDP):
          use_r_goal           : include r_budget + r_calorie + r_variety shaping
          use_terminal_reward  : add ±5..±50 terminal bonus at episode end
          use_fatigue_dynamics : per-category fatigue accumulates and decays
        """
        self.weekly_budget        = weekly_budget
        self.weekly_calories      = weekly_calories
        self.user_profile         = user_profile or UserProfile()
        self.use_r_goal           = use_r_goal
        self.use_terminal_reward  = use_terminal_reward
        self.use_fatigue_dynamics = use_fatigue_dynamics
        self.reset()

    # ── reset ──
    def reset(self):
        self.step_idx              = 0
        self.budget_spent          = 0.0
        self.calories_eaten        = 0
        self.fatigue               = [0.0] * N_CAT
        self.consec_churns         = 0
        self.meal_history          = []          # list of category indices
        self.pref_weights          = list(self.user_profile.init_prefs)
        self.avg_price_pref        = 12.0
        self.log                   = []          # list of step dicts
        self.done                  = False
        self.last_accept_clock_h   = None        # absolute hour of last accepted meal
        return self._get_state()

    # absolute clock hour for a given step index (day*24 + meal_hour)
    def _clock_hour(self, step_idx):
        d   = step_idx // N_MEALS_DAY
        mt  = step_idx %  N_MEALS_DAY
        return d * 24 + MEAL_HOURS[mt]

    def _time_since_last(self):
        """Hours since last accepted meal (0 if first meal, capped at 24)."""
        if self.last_accept_clock_h is None:
            return 12.0   # neutral default before any acceptance
        now_h = self._clock_hour(self.step_idx)
        return float(min(max(now_h - self.last_accept_clock_h, 0), 24))

    # ── step ──
    def step(self, action_idx):
        assert not self.done
        meal        = MEALS[action_idx]
        day_idx     = self.step_idx // N_MEALS_DAY
        meal_t_idx  = self.step_idx % N_MEALS_DAY
        is_weekend  = day_idx >= 5

        # filter: meal must serve this meal time
        mt = MEAL_TIMES[meal_t_idx]
        if mt not in meal["meal_types"]:
            # invalid action → force churn equivalent but mild
            reward = -3.0
            response = "invalid"
            self._advance(meal, response, reward, day_idx, meal_t_idx)
            return self._get_state(), reward, self.done, self._last_log()

        # user response
        response, user_reward = self.user_profile.respond(
            meal, self.fatigue, self.pref_weights,
            self.budget_spent, self.weekly_budget,
            self.calories_eaten, self.weekly_calories,
            self.step_idx
        )

        # reward components
        r_accept  = {"accept": 5.0, "accept_browse": 3.0,
                     "reject": -2.0, "churn": -10.0}[response]

        meals_left = MAX_STEPS - self.step_idx
        budget_pace  = (self.weekly_budget   - self.budget_spent)   / max(meals_left, 1)
        calorie_pace = (self.weekly_calories - self.calories_eaten) / max(meals_left, 1)

        r_budget  = 3.0 if meal["price"]    <= budget_pace  * 1.0 else \
                    1.0 if meal["price"]    <= budget_pace  * 1.2 else 0.0
        r_calorie = 2.0 if meal["calories"] <= calorie_pace * 1.0 else \
                    1.0 if meal["calories"] <= calorie_pace * 1.15 else 0.0

        cat_idx   = CATEGORIES.index(meal["category"])
        recent3   = self.meal_history[-3:]
        r_variety = 1.0 if cat_idx not in recent3 else \
                   -2.0 if len(recent3) == 3 and all(c == cat_idx for c in recent3) else 0.0

        r_goal  = 0.4 * r_budget + 0.3 * r_calorie + 0.3 * r_variety
        reward  = r_accept + (r_goal if self.use_r_goal else 0.0)

        # state updates
        if response in ("accept", "accept_browse"):
            self.budget_spent   += meal["price"]
            self.calories_eaten += meal["calories"]
            self.meal_history.append(cat_idx)
            self.last_accept_clock_h = self._clock_hour(self.step_idx)
            # update pref weights (EMA)
            alpha = 0.1
            self.pref_weights[cat_idx] = (
                (1 - alpha) * self.pref_weights[cat_idx] + alpha * 1.0
            )
            self.avg_price_pref = (
                0.9 * self.avg_price_pref + 0.1 * meal["price"]
            )
            self.consec_churns = 0
        else:
            self.consec_churns += 1
        self.user_profile.record_feedback(meal, response)

        # fatigue dynamics (decay + update on accept) — disabled in ablation
        if self.use_fatigue_dynamics:
            for i in range(N_CAT):
                self.fatigue[i] *= FATIGUE_DECAY
            if response in ("accept", "accept_browse"):
                self.fatigue[cat_idx] = min(self.fatigue[cat_idx] + 1.0, 3.0)

        # terminal reward
        r_terminal = 0.0
        self._advance(meal, response, reward, day_idx, meal_t_idx,
                      r_accept=r_accept, r_goal=r_goal,
                      r_budget=r_budget, r_calorie=r_calorie, r_variety=r_variety)

        self.step_idx += 1
        if self.step_idx >= MAX_STEPS or self.consec_churns >= CHURN_LIMIT:
            self.done = True
            budget_ok  = self.budget_spent   <= self.weekly_budget
            calorie_ok = self.calories_eaten <= self.weekly_calories
            if budget_ok and calorie_ok and self.consec_churns == 0:
                r_terminal = 50.0
            elif budget_ok or calorie_ok:
                r_terminal = 20.0
            elif self.consec_churns < CHURN_LIMIT:
                r_terminal = 5.0
            else:
                r_terminal = -30.0
            if not self.use_terminal_reward:
                r_terminal = 0.0
            reward += r_terminal
            self.log[-1]["r_terminal"] = r_terminal
            self.log[-1]["reward"]    += r_terminal

        return self._get_state(), reward, self.done, self._last_log()

    # ── helpers ──
    def _get_state(self):
        day_idx   = min(self.step_idx // N_MEALS_DAY, N_DAYS - 1)
        mt_idx    = self.step_idx % N_MEALS_DAY
        days_rem  = max(N_DAYS - day_idx, 1)
        meals_rem = max(MAX_STEPS - self.step_idx, 1)
        return build_state(
            budget_rem   = max(1 - self.budget_spent / self.weekly_budget, 0),
            cal_rem      = max(1 - self.calories_eaten / self.weekly_calories, 0),
            days_rem     = days_rem,
            meals_rem    = meals_rem,
            fatigue      = self.fatigue,
            meal_time_idx= mt_idx,
            day_idx      = day_idx,
            time_since_last = self._time_since_last(),
            is_weekend   = day_idx >= 5,
            pref_weights = self.pref_weights,
            avg_price_pref = self.avg_price_pref,
            consec_churns  = self.consec_churns,
        )

    def _advance(self, meal, response, reward, day_idx, mt_idx, **kwargs):
        self.log.append({
            "step":       self.step_idx,
            "day":        DAYS[min(day_idx, 6)],
            "meal_time":  MEAL_TIMES[mt_idx],
            "meal":       meal["name"],
            "category":   meal["category"],
            "emoji":      meal["emoji"],
            "price":      meal["price"],
            "calories":   meal["calories"],
            "tags":       list(meal.get("tags", [])),
            "response":   response,
            "reward":     reward,
            "r_terminal": 0.0,
            "fatigue":    list(self.fatigue),
            "budget_spent":   self.budget_spent,
            "calories_eaten": self.calories_eaten,
            "consec_churns":  self.consec_churns,
            **kwargs,
        })

    def _last_log(self):
        return self.log[-1] if self.log else {}

    def get_valid_actions(self):
        mt = MEAL_TIMES[self.step_idx % N_MEALS_DAY]
        meal_time_valid = [i for i, m in enumerate(MEALS) if mt in m["meal_types"]]
        constrained = [
            i for i in meal_time_valid
            if self.user_profile.accepts_meal(MEALS[i])
        ]
        return constrained or meal_time_valid

    def summary(self):
        total_reward = sum(l["reward"] for l in self.log)
        budget_ok    = self.budget_spent <= self.weekly_budget
        calorie_ok   = self.calories_eaten <= self.weekly_calories
        churns       = sum(1 for l in self.log if l["response"] == "churn")
        return {
            "total_reward":   total_reward,
            "budget_ok":      budget_ok,
            "calorie_ok":     calorie_ok,
            "budget_spent":   self.budget_spent,
            "calories_eaten": self.calories_eaten,
            "churns":         churns,
            "steps":          self.step_idx,
        }


# ── User Profile / Simulator ───────────────────────────────────────────────
class UserProfile:
    """Synthetic user whose acceptance probability depends on context."""

    def __init__(self, preferred_cat=None, budget_sensitivity=0.5,
                 calorie_sensitivity=0.3, fatigue_sensitivity=0.6,
                 noise=0.15, preference_drift_at=None, drift_to=None,
                 preferred_tags=None, avoided_tags=None, disliked_cats=None,
                 max_price=None, max_calories_per_meal=None):
        self.preferred_cat        = preferred_cat or "Japanese"
        self.budget_sensitivity   = budget_sensitivity
        self.calorie_sensitivity  = calorie_sensitivity
        self.fatigue_sensitivity  = fatigue_sensitivity
        self.noise                = noise
        self.preference_drift_at  = preference_drift_at
        self.drift_to             = drift_to
        self.step_count           = 0
        self.preferred_tags       = set(preferred_tags or [])
        self.avoided_tags         = set(avoided_tags or [])
        self.disliked_cats        = set(disliked_cats or [])
        self.max_price            = max_price
        self.max_calories_per_meal = max_calories_per_meal
        self.tag_affinity         = {tag: 0.0 for meal in MEALS for tag in meal.get("tags", [])}

        n = N_CAT
        idx = CATEGORIES.index(self.preferred_cat)
        prefs = np.ones(n) * 0.3
        prefs[idx] = 1.0
        for cat in self.disliked_cats:
            if cat in CATEGORIES and cat != self.preferred_cat:
                prefs[CATEGORIES.index(cat)] = 0.05
        prefs = prefs / prefs.sum()
        self.init_prefs = prefs

    def accepts_meal(self, meal):
        """Hard product constraints become the RL action mask."""
        tags = set(meal.get("tags", []))
        if meal["category"] in self.disliked_cats:
            return False
        if self.max_price is not None and meal["price"] > self.max_price:
            return False
        if (self.max_calories_per_meal is not None
                and meal["calories"] > self.max_calories_per_meal):
            return False
        if tags & self.avoided_tags:
            return False
        return True

    def record_feedback(self, meal, response):
        """Session-level preference learning from the same env feedback loop."""
        delta = {
            "accept": 0.10,
            "accept_browse": 0.05,
            "reject": -0.06,
            "churn": -0.12,
            "invalid": -0.08,
        }.get(response, 0.0)
        for tag in meal.get("tags", []):
            self.tag_affinity[tag] = float(np.clip(
                self.tag_affinity.get(tag, 0.0) + delta,
                -0.5, 0.5
            ))

    def respond(self, meal, fatigue, pref_weights,
                budget_spent, weekly_budget,
                calories_eaten, weekly_calories,
                step_idx):
        # optional preference drift
        if self.preference_drift_at and step_idx == self.preference_drift_at:
            self.preferred_cat = self.drift_to or "Light"

        cat_idx = CATEGORIES.index(meal["category"])

        # base preference — always positive for known meals
        base = 1.5

        # strong boost for preferred category
        if meal["category"] == self.preferred_cat:
            base += 2.0
        else:
            base += pref_weights[cat_idx] * 2.0  # grows as agent learns

        tags = set(meal.get("tags", []))
        base += 0.35 * len(tags & self.preferred_tags)
        base -= 1.8 * len(tags & self.avoided_tags)
        base += sum(self.tag_affinity.get(tag, 0.0) for tag in tags)
        if meal["category"] in self.disliked_cats:
            base -= 2.0

        # fatigue penalty: kicks in hard above 0.8
        if fatigue[cat_idx] > 0.8:
            base -= (fatigue[cat_idx] - 0.8) * self.fatigue_sensitivity * 3.0

        # budget fit
        meals_left  = max(MAX_STEPS - step_idx, 1)
        budget_left = weekly_budget - budget_spent
        pace        = budget_left / meals_left
        if meal["price"] > pace * 1.5:
            base -= self.budget_sensitivity * 1.5
        elif meal["price"] <= pace:
            base += 0.3
        if self.max_price is not None and meal["price"] <= self.max_price * 0.85:
            base += 0.2

        # calorie fit
        cal_left = weekly_calories - calories_eaten
        cal_pace = cal_left / meals_left
        if meal["calories"] > cal_pace * 1.4:
            base -= self.calorie_sensitivity * 1.0
        if (self.max_calories_per_meal is not None
                and meal["calories"] <= self.max_calories_per_meal * 0.85):
            base += 0.2

        # add noise
        score = base + np.random.normal(0, self.noise)

        self.step_count += 1

        # thresholds — calibrated so preferred meal ~80% accept,
        # non-preferred ~50% reject, churns only on repeated bad recs
        if score >= 2.0:
            return "accept", score
        elif score >= 1.2:
            return "accept_browse", score
        elif score >= 0.2:
            return "reject", score
        else:
            return "churn", score
