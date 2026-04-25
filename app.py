"""
TasteFlow — Streamlit Demo App
Two modes:
  📊 Watch the Agent Learn  — animated training + before/after comparison
  🎮 Be the User            — interactive single episode (mobile-app mockup)

Run:
  cd tasteflow_app
  python train.py       # one-time training (~30s)
  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from env import (TasteFlowEnv, UserProfile, CATEGORIES, MEALS,
                 MEAL_TIMES, DAYS)
from agents import (GreedyAgent, EpsilonGreedyAgent, PPOAgent, LagrangianPPOAgent)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TasteFlow RL Demo",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light Hi-Fi CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg:        #F7F8FA;
    --surface:   #FFFFFF;
    --surface-2: #F1F5F9;
    --border:    #E2E8F0;
    --text:      #0F172A;
    --text-mute: #475569;
    --primary:   #0F766E;
    --primary-50:#F0FDFA;
    --accept:    #16A34A;
    --browse:    #CA8A04;
    --reject:    #EA580C;
    --churn:     #DC2626;
  }

  /* ── Base canvas ── */
  [data-testid="stAppViewContainer"]{background:var(--bg);}
  [data-testid="stSidebar"]{background:var(--surface); border-right:1px solid var(--border);}
  [data-testid="stHeader"]{background:transparent;}
  [data-testid="block-container"]{padding-top:1.4rem; padding-bottom:2rem;}

  /* ── Typography (fix #6 banner H1) ── */
  h1{font-size:1.55rem !important; font-weight:700 !important;
     margin:.2rem 0 .15rem !important; letter-spacing:-.01em; color:var(--text) !important;}
  h2{font-size:1.18rem !important; font-weight:700 !important; color:var(--text) !important;
     margin:.4rem 0 .3rem !important;}
  h3{font-size:1.0rem !important; font-weight:700 !important; color:var(--text) !important;}
  h4{font-size:.92rem !important; color:var(--text) !important; font-weight:700 !important;}
  h5{font-size:.85rem !important; color:var(--text) !important; font-weight:700 !important;}
  .stMarkdown p, .stMarkdown li{color:var(--text);}
  .page-sub{color:var(--text-mute); font-size:.92rem; margin:-.1rem 0 .6rem 0;}

  /* ── Tabs (fix #1 horizontal scroll, #7 contrast) ── */
  .stTabs [data-baseweb="tab-list"]{
    gap:2px;
    overflow-x:auto !important;
    flex-wrap:nowrap !important;
    border-bottom:1px solid var(--border);
    scrollbar-width:thin;
  }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{height:6px;}
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb{
    background:#CBD5E1; border-radius:3px;
  }
  .stTabs [data-baseweb="tab"]{
    color:#334155 !important; font-weight:600;
    padding:10px 14px; background:transparent;
    border-radius:8px 8px 0 0; white-space:nowrap;
  }
  .stTabs [aria-selected="true"]{
    color:var(--primary) !important;
    background:var(--surface);
    border-bottom:2px solid var(--primary) !important;
  }

  /* ── Cards ── */
  .card{
    background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:14px 16px;
    box-shadow:0 1px 2px rgba(15,23,42,.04);
  }
  .meal-card{
    background:linear-gradient(135deg,#FFFFFF,var(--primary-50));
    border:1px solid #99F6E4; border-radius:14px; padding:18px 22px;
  }

  /* ── Key Takeaways (fix #9) ── */
  .takeaway-row{
    display:flex; flex-direction:column; gap:2px;
    padding:10px 14px; margin:6px 0;
    background:var(--surface); border:1px solid var(--border);
    border-left-width:3px; border-radius:0 10px 10px 0;
  }
  .takeaway-head{display:flex; align-items:baseline; gap:8px;}
  .takeaway-name{font-weight:800; font-size:.95rem;}
  .takeaway-meta{color:var(--text-mute); font-size:.78rem;}
  .takeaway-body{color:var(--text) !important; opacity:1 !important;
                 font-size:.88rem; line-height:1.5; margin-top:2px;}

  /* ── Metric containers ── */
  div[data-testid="metric-container"]{
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:10px 12px;
  }

  /* ── Generic buttons (override old teal default) ── */
  .stButton > button{
    border:1px solid var(--border) !important;
    background:var(--surface) !important; color:var(--text) !important;
    border-radius:10px !important; font-weight:600;
    transition:filter .15s ease, transform .15s ease;
  }
  .stButton > button:hover{
    filter:brightness(.97); transform:translateY(-1px);
    border-color:#CBD5E1 !important;
  }
  /* Semantic action buttons (fix #4) */
  .btn-accept .stButton > button{background:var(--accept) !important; color:#fff !important; border-color:transparent !important;}
  .btn-browse .stButton > button{background:var(--browse) !important; color:#fff !important; border-color:transparent !important;}
  .btn-reject .stButton > button{background:var(--reject) !important; color:#fff !important; border-color:transparent !important;}
  .btn-churn  .stButton > button{background:var(--churn)  !important; color:#fff !important; border-color:transparent !important;}
  .btn-accept .stButton > button:hover,
  .btn-browse .stButton > button:hover,
  .btn-reject .stButton > button:hover,
  .btn-churn  .stButton > button:hover{filter:brightness(1.08); transform:translateY(-1px);}

  /* ── Phone mockup (fix #2/#3 inside, plus mobile-app feel) ── */
  /* Outer dark bezel applied to the column that contains a .phone-marker */
  [data-testid="stColumn"]:has(.phone-marker){
    background:#0F172A !important;
    border-radius:38px !important;
    padding:16px 12px 26px !important;
    box-shadow:0 24px 60px rgba(15,23,42,.20), 0 0 0 1px #1E293B inset;
    position:relative;
    margin-top:6px;
  }
  /* Notch */
  [data-testid="stColumn"]:has(.phone-marker)::before{
    content:""; position:absolute; top:18px; left:50%; transform:translateX(-50%);
    width:118px; height:22px; background:#0F172A; border-radius:0 0 14px 14px;
    z-index:5;
  }
  /* Home indicator */
  [data-testid="stColumn"]:has(.phone-marker)::after{
    content:""; position:absolute; bottom:10px; left:50%; transform:translateX(-50%);
    width:96px; height:4px; background:#475569; border-radius:2px;
  }
  /* The inner white "screen" — the column's vertical block */
  [data-testid="stColumn"]:has(.phone-marker) > [data-testid="stVerticalBlock"]{
    background:#FFFFFF !important;
    border-radius:28px !important;
    padding:42px 12px 18px !important;
    min-height:620px;
  }
  .phone-marker{display:none;}

  /* Phone status bar */
  .phone-statusbar{
    display:flex; gap:6px; align-items:center; justify-content:space-between;
    padding:7px 10px; background:#F8FAFC; border:1px solid var(--border);
    border-radius:10px; margin-bottom:10px;
    font-size:.72rem; color:var(--text-mute);
  }
  .phone-statusbar b{color:var(--text); font-size:.78rem;}

  /* Phone progress row */
  .phone-progress{display:flex; gap:8px; margin-bottom:12px;}
  .phone-progress > div{
    flex:1; padding:8px 10px; background:#F8FAFC;
    border:1px solid var(--border); border-radius:10px; font-size:.72rem;
    color:var(--text-mute); white-space:nowrap;
  }
  .phone-progress b{color:var(--text); font-size:.85rem;}
  .phone-progress .bar{height:4px; background:#E2E8F0; border-radius:2px; margin-top:5px;}
  .phone-progress .bar > span{display:block; height:4px; border-radius:2px;}

  /* Phone meal recommendation card */
  .phone-meal{
    background:linear-gradient(135deg,var(--primary-50),#FFFFFF);
    border:1px solid #99F6E4; border-radius:16px; padding:14px;
    margin-bottom:12px;
  }
  .pill{
    display:inline-block; padding:3px 9px; border-radius:999px;
    font-size:.72rem; font-weight:600;
  }
  .pill-money{background:#ECFDF5; color:#15803D;}
  .pill-cal  {background:#F5F3FF; color:#6D28D9;}
  .pill-fa-ok{background:var(--primary-50); color:var(--primary);}
  .pill-fa-hi{background:#FEF2F2; color:var(--churn);}

  /* Phone history list */
  .phone-history{
    margin-top:6px; border-top:1px dashed var(--border); padding-top:8px;
  }
  .phone-history-row{
    display:flex; gap:8px; align-items:center;
    padding:6px 0; font-size:.78rem; color:var(--text);
  }
  .phone-history-row .name{
    flex:1; min-width:0; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap;
  }

  /* Buttons inside the phone — make them more compact */
  [data-testid="stColumn"]:has(.phone-marker) .stButton > button{
    padding:.45rem .25rem !important;
    font-size:.85rem !important;
    border-radius:12px !important;
  }

  /* Keep the mock phone usable before Streamlit's columns naturally collapse. */
  [data-testid="stHorizontalBlock"]:has(.phone-marker){
    align-items:flex-start;
  }
  @media (max-width: 1120px){
    [data-testid="stHorizontalBlock"]:has(.phone-marker){
      flex-direction:column !important;
      gap:1rem !important;
    }
    [data-testid="stHorizontalBlock"]:has(.phone-marker) > [data-testid="stColumn"]{
      width:100% !important;
      flex:1 1 auto !important;
    }
    [data-testid="stColumn"]:has(.phone-marker){
      width:min(100%, 330px) !important;
      max-width:330px !important;
      margin:6px auto 18px !important;
      box-sizing:border-box;
    }
  }
  @media (max-width: 420px){
    [data-testid="block-container"]{padding-left:.75rem; padding-right:.75rem;}
    [data-testid="stColumn"]:has(.phone-marker){
      padding:12px 9px 22px !important;
      border-radius:32px !important;
    }
    [data-testid="stColumn"]:has(.phone-marker)::before{
      top:14px; width:104px; height:18px;
    }
    [data-testid="stColumn"]:has(.phone-marker)::after{
      bottom:8px; width:84px;
    }
    [data-testid="stColumn"]:has(.phone-marker) > [data-testid="stVerticalBlock"]{
      border-radius:24px !important;
      padding:34px 10px 16px !important;
      min-height:560px;
    }
    .phone-statusbar{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:5px 8px;
      padding:7px 9px;
      font-size:.68rem;
    }
    .phone-statusbar span:last-child{
      grid-column:1 / -1;
    }
    .phone-progress{
      display:grid;
      grid-template-columns:1fr;
      gap:7px;
    }
    .phone-progress > div{
      padding:7px 9px;
      font-size:.7rem;
    }
    .phone-meal{
      padding:12px;
    }
    .phone-meal .meal-main{
      flex-direction:column;
      align-items:flex-start !important;
      gap:6px !important;
    }
    .phone-meal .meal-emoji{
      font-size:2.2em !important;
    }
    .phone-meal .meal-name{
      font-size:.98rem !important;
      overflow-wrap:normal;
      word-break:normal;
    }
    .pill{
      font-size:.68rem;
      padding:3px 8px;
    }
    [data-testid="stColumn"]:has(.phone-marker) .stButton > button{
      font-size:.78rem !important;
      min-height:2.25rem;
    }
  }

  /* ── Sticky comparison table (fix #8) ── */
  .compare-table-wrap{
    overflow-x:auto; border:1px solid var(--border);
    border-radius:10px; background:var(--surface); margin-top:6px;
  }
  .compare-table-wrap table{border-collapse:separate; border-spacing:0; width:100%;
    font-family:-apple-system, system-ui, sans-serif; font-size:.86rem;}
  .compare-table-wrap th, .compare-table-wrap td{padding:9px 14px;}
  .compare-table-wrap thead th{
    background:var(--surface-2); color:var(--text-mute); font-weight:700;
    border-bottom:1px solid var(--border); text-align:center;
  }
  .compare-table-wrap thead th.metric-col,
  .compare-table-wrap tbody td.metric-col{
    position:sticky; left:0; z-index:3;
    background:var(--surface) !important;
    border-right:1px solid var(--border);
    text-align:left; min-width:160px;
    color:var(--text-mute); font-weight:600;
  }
  .compare-table-wrap thead th.metric-col{background:var(--surface-2) !important;}
  .compare-table-wrap tbody tr{border-top:1px solid var(--border);}
  .compare-table-wrap tbody td{text-align:center;}

  /* Equation block on light bg */
  .eq-card{
    background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:14px 16px;
    font-family:"SF Mono", Menlo, monospace; font-size:.84rem;
    line-height:1.7; color:var(--text);
  }

  /* Mode banner above tabs */
  .mode-banner{
    background:linear-gradient(135deg, var(--primary-50), #FFFFFF);
    border:1px solid #99F6E4; border-radius:12px;
    padding:10px 14px; margin:6px 0 14px; font-size:.85rem; color:var(--text);
  }
</style>
""", unsafe_allow_html=True)


# ── Palette (tuned for light bg, WCAG AA where possible) ───────────────────
AGENT_COLORS = {
    "Random":      "#94A3B8",
    "Greedy":      "#E11D48",
    "ε-Greedy":    "#F97316",
    "Rule-Based":  "#EA580C",
    "LinUCB":      "#CA8A04",
    "PPO":         "#0F766E",
    "PPO-L":       "#7C3AED",
}
RESPONSE_EMOJI = {
    "accept": "✅", "accept_browse": "👍",
    "reject": "😐", "churn": "💨", "invalid": "❌",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#0F172A", family="-apple-system, system-ui, sans-serif", size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#E2E8F0", borderwidth=1),
)
GRID = dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", linecolor="#CBD5E1")


def style_axes(fig, n_subplots=1, axis_overrides=None):
    """Apply light-theme axis grid colors uniformly."""
    over = axis_overrides or {}
    if n_subplots == 1:
        fig.update_xaxes(**GRID, **over.get("x", {}))
        fig.update_yaxes(**GRID, **over.get("y", {}))
    return fig


# ── Load results ───────────────────────────────────────────────────────────
@st.cache_resource
def load_results():
    path = os.path.join(os.path.dirname(__file__), "tasteflow_results.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def make_trained_ppo(weights):
    agent = PPOAgent()
    for k, v in weights.items():
        setattr(agent, k, v)
    return agent


def make_trained_ppol(weights):
    agent = LagrangianPPOAgent()
    for k, v in weights.items():
        setattr(agent, k, v)
    return agent


ALL_MEAL_TAGS = sorted({tag for meal in MEALS for tag in meal.get("tags", [])})
MEAL_EMOJI_BY_NAME = {meal["name"]: meal["emoji"] for meal in MEALS}


def display_meal_emoji(record):
    """Use the current meal catalog so cached demo logs do not show stale icons."""
    return MEAL_EMOJI_BY_NAME.get(
        record.get("meal") or record.get("name"),
        record.get("emoji", "🍽️"),
    )


def get_policy_probs(agent, state, valid_actions):
    """Return an action-probability vector for both neural and baseline agents."""
    if hasattr(agent, "get_action_probs"):
        return agent.get_action_probs(state, valid_actions)

    probs = np.zeros(len(MEALS), dtype=float)
    if hasattr(agent, "Q"):
        q_vals = np.array([agent.Q[i] for i in valid_actions], dtype=float)
        q_vals = q_vals - q_vals.max()
        exp_q = np.exp(q_vals)
        base = exp_q / max(exp_q.sum(), 1e-8)
        if hasattr(agent, "epsilon"):
            base = (1 - agent.epsilon) * base + agent.epsilon / len(valid_actions)
        for idx, action in enumerate(valid_actions):
            probs[action] = base[idx]
    else:
        for action in valid_actions:
            probs[action] = 1.0 / len(valid_actions)
    return probs


def build_user_profile(config):
    intent = config["intent"]
    budget_sensitivity = 0.5
    calorie_sensitivity = 0.3
    preferred_tags = set(config["preferred_tags"])

    if intent == "Save money":
        budget_sensitivity = 0.9
        preferred_tags.add("budget")
    elif intent == "Eat healthier":
        calorie_sensitivity = 0.75
        preferred_tags.update(["healthy", "light", "balanced"])
    elif intent == "Quick meal":
        preferred_tags.add("quick")
    elif intent == "High protein":
        preferred_tags.add("high-protein")
    elif intent == "Comfort food":
        preferred_tags.add("comfort")

    return UserProfile(
        preferred_cat=config["preferred_cat"],
        budget_sensitivity=budget_sensitivity,
        calorie_sensitivity=calorie_sensitivity,
        fatigue_sensitivity=0.6,
        noise=0.08,
        preferred_tags=sorted(preferred_tags),
        avoided_tags=config["avoided_tags"],
        disliked_cats=config["disliked_cats"],
        max_price=config["max_price"],
        max_calories_per_meal=config["max_calories_per_meal"],
    )


def explain_recommendation(meal, env, prob):
    tags = ", ".join(meal.get("tags", [])[:3])
    meals_left = max(21 - env.step_idx, 1)
    budget_pace = (env.weekly_budget - env.budget_spent) / meals_left
    calorie_pace = (env.weekly_calories - env.calories_eaten) / meals_left
    cat_idx = CATEGORIES.index(meal["category"])
    reasons = [
        f"policy confidence {prob * 100:.1f}%",
        f"{meal['category']} matches your current preference vector",
    ]
    if meal["price"] <= budget_pace:
        reasons.append("within today's budget pace")
    if meal["calories"] <= calorie_pace:
        reasons.append("within calorie pace")
    if env.fatigue[cat_idx] <= 1.0:
        reasons.append("low category fatigue")
    if tags:
        reasons.append(f"tags: {tags}")
    return " · ".join(reasons)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍜 TasteFlow")
    st.caption("RL-powered food decisions")
    st.divider()

    mode = st.radio(
        "Demo Mode",
        ["📊 Watch the Agent Learn", "🎮 Be the User"],
        index=0,
    )

    st.divider()
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
**TasteFlow** trains **PPO** and **PPO-Lagrangian (PPO-L)** to learn weekly food preferences.

- **State (24-dim):** budget, calories, fatigue, meal time, prefs
- **Reward:** accept +5, reject −2, churn −10, weekly goals +50
- **Episode:** 21 meals (3/day × 7 days)
        """)

    st.divider()
    st.markdown("**Current Session**")
    if mode == "🎮 Be the User" and "game_env" in st.session_state:
        env_s = st.session_state.game_env
        cur_day = min(env_s.step_idx // len(MEAL_TIMES), 6) + 1
        budget_left_s = max(env_s.weekly_budget - env_s.budget_spent, 0)
        s1, s2 = st.columns(2)
        s1.metric("Reward", f"{st.session_state.cumulative_r:+.1f}")
        s2.metric("Day", f"{cur_day} / 7")
        st.metric("Budget left", f"${budget_left_s:.0f}")
    else:
        st.caption("Stats appear after entering 'Be the User' mode.")


# ═══════════════════════════════════════════════════════════════
#  MODE 1: WATCH THE AGENT LEARN
# ═══════════════════════════════════════════════════════════════
if mode == "📊 Watch the Agent Learn":

    st.markdown("## 📊 Watch the Agent Learn")
    st.markdown(
        '<p class="page-sub">See how PPO improves over 800 training episodes — and compare against baselines.</p>',
        unsafe_allow_html=True,
    )

    data = load_results()

    if data is None:
        st.warning("⚠️ No training data found. Run `python train.py` first (takes ~30 seconds).")
        if st.button("🚀 Run Training Now"):
            with st.spinner("Training..."):
                import subprocess
                subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "train.py")],
                    capture_output=True, text=True,
                )
            st.rerun()
    else:
        ppo_curve  = data["ppo_curve"]
        results    = data["results"]
        log_before = data["log_before"]
        log_after  = data["log_after"]

        st.caption("← Scroll horizontally to see all tabs →")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Learning Curve", "🏆 Agent Comparison",
            "🔄 Before vs After", "🧠 State & Reward Breakdown",
            "⚖️ PPO-L: Constraint Analysis",
        ])

        # ── TAB 1: Learning Curve ──
        with tab1:
            st.markdown("### Learning Curves — PPO vs PPO-Lagrangian (800 Episodes)")
            st.markdown(
                "Both agents learn from scratch. **PPO** maximises total reward "
                "(including soft goal bonuses). **PPO-L** maximises *only* acceptance "
                "reward while the Lagrange multipliers handle constraints separately — "
                "watch how λ_budget grows as the agent learns to pace spending."
            )

            ppol_curve = data.get("ppol_curve", data.get("cpo_curve", []))
            lambda_b   = data.get("lambda_b_curve", [])
            lambda_c   = data.get("lambda_c_curve", [])

            eps_ppo   = [x[0] for x in ppo_curve]
            rews_ppo  = [x[1] for x in ppo_curve]
            eps_ppol  = [x[0] for x in ppol_curve]
            rews_ppol = [x[1] for x in ppol_curve]

            fig = make_subplots(rows=1, cols=2,
                subplot_titles=("Episode Reward (rolling mean 20)", "PPO-L Lagrange Multipliers λ"))

            fig.add_trace(go.Scatter(x=eps_ppo, y=rews_ppo, mode="lines",
                line=dict(color=AGENT_COLORS["PPO"], width=2.5), name="PPO"), row=1, col=1)
            fig.add_trace(go.Scatter(x=eps_ppol, y=rews_ppol, mode="lines",
                line=dict(color=AGENT_COLORS["PPO-L"], width=2.5), name="PPO-L"), row=1, col=1)
            fig.add_vline(x=100, line_dash="dash", line_color="#DC2626",
                line_width=1.2, row=1, col=1)
            fig.add_annotation(x=120, y=min(rews_ppo)*0.85,
                text="Preference drift", showarrow=False,
                font=dict(color="#DC2626", size=10), row=1, col=1)

            if lambda_b:
                eps_l  = [x[0] for x in lambda_b]
                lb_v   = [x[1] for x in lambda_b]
                lc_v   = [x[1] for x in lambda_c]
                fig.add_trace(go.Scatter(x=eps_l, y=lb_v, mode="lines",
                    line=dict(color="#EA580C", width=2), name="λ_budget"), row=1, col=2)
                fig.add_trace(go.Scatter(x=eps_l, y=lc_v, mode="lines",
                    line=dict(color="#15803D", width=2), name="λ_calorie"), row=1, col=2)
                fig.add_annotation(
                    x=eps_l[-1]*0.6, y=max(lb_v)*0.85,
                    text="λ grows when constraint<br>is violated → auto-penalty",
                    showarrow=False,
                    font=dict(color="#EA580C", size=10),
                    row=1, col=2,
                )

            fig.update_layout(**PLOTLY_LAYOUT, height=380,
                margin=dict(t=50, b=40, l=60, r=20))
            for c in [1, 2]:
                fig.update_xaxes(title="Episode", **GRID, row=1, col=c)
                fig.update_yaxes(**GRID, row=1, col=c)
            fig.update_yaxes(title="Reward", row=1, col=1)
            fig.update_yaxes(title="λ value", row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(4)
            cols[0].metric("PPO ep10",   f"{rews_ppo[0]:+.1f}")
            cols[1].metric("PPO final",  f"{rews_ppo[-1]:+.1f}",
                           delta=f"{rews_ppo[-1]-rews_ppo[0]:+.1f}")
            cols[2].metric("PPO-L ep10",   f"{rews_ppol[0]:+.1f}" if rews_ppol else "—")
            cols[3].metric("PPO-L final",  f"{rews_ppol[-1]:+.1f}" if rews_ppol else "—",
                           delta=f"{rews_ppol[-1]-rews_ppol[0]:+.1f}" if rews_ppol else None)

        # ── TAB 2: Agent Comparison ──
        with tab2:
            st.markdown("### 🏆 Agent Comparison — 50 Evaluation Episodes")
            st.markdown(
                "All agents evaluated on the same synthetic user. "
                "**Random** is the baseline. % gains show improvement relative to it."
            )

            agent_names = list(results.keys())
            baseline    = "Random"
            base        = results[baseline]
            colors      = [AGENT_COLORS.get(n, "#94A3B8") for n in agent_names]

            def gain(metric, name, higher_is_better=True):
                bv = base[metric]
                av = results[name][metric]
                if name == baseline or abs(bv) < 1e-6:
                    return None
                raw = (av - bv) / abs(bv) * 100
                return raw if higher_is_better else -raw

            # ── 1. Reward bar chart ──
            st.markdown("#### Reward vs Baseline")
            rews = [results[n]["mean_reward"] for n in agent_names]
            stds = [results[n]["std_reward"]  for n in agent_names]

            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=agent_names, y=rews,
                error_y=dict(type="data", array=stds, color="rgba(15,23,42,0.25)"),
                marker_color=colors,
                text=[f"{v:+.1f}" for v in rews],
                textposition=["inside" if n == baseline else "outside" for n in agent_names],
                insidetextanchor="end",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Reward: %{y:.1f}<extra></extra>",
            ))
            fig_r.add_hline(
                y=base["mean_reward"],
                line_dash="dash", line_color="#94A3B8", line_width=1.5,
            )
            fig_r.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                xaxis=dict(**GRID),
                yaxis=dict(title="Mean Episode Reward", **GRID, range=[0, max(rews)*1.25]),
                margin=dict(t=20, b=20, l=60, r=20),
            )
            st.plotly_chart(fig_r, use_container_width=True)
            st.caption(f"Dashed line marks the Random baseline ({base['mean_reward']:.1f}).")

            # ── 2. Full comparison table — sticky first col + horizontal scroll ──
            st.markdown("#### Full Comparison Table — % Gain vs Random Baseline")
            st.caption("→ Scroll horizontally; the **Metric** column stays pinned.")

            METRICS = [
                ("mean_reward",       "Reward",           True,  "{:+.1f}"),
                ("budget_pct",        "Budget Met %",     True,  "{:.0f}%"),
                ("calorie_pct",       "Calorie Met %",    True,  "{:.0f}%"),
                ("constraint_ok_pct", "Both Goals Met %", True,  "{:.0f}%"),
                ("churn_rate",        "Churn Rate %",     False, "{:.1f}%"),
            ]

            header_html = '<div class="compare-table-wrap"><table><thead><tr>'
            header_html += '<th class="metric-col">Metric</th>'
            for n in agent_names:
                col = AGENT_COLORS.get(n, "#0F172A")
                header_html += (
                    f'<th style="color:{col};min-width:120px">{n}</th>'
                )
            header_html += "</tr></thead><tbody>"

            rows_html = ""
            for metric_key, metric_label, higher_better, fmt in METRICS:
                rows_html += "<tr>"
                rows_html += f'<td class="metric-col">{metric_label}</td>'
                vals     = [results[n][metric_key] for n in agent_names]
                best_val = max(vals) if higher_better else min(vals)
                for n in agent_names:
                    v = results[n][metric_key]
                    g = gain(metric_key, n, higher_better)
                    is_best  = abs(v - best_val) < 0.01
                    cell_bg  = "rgba(15,118,110,0.08)" if is_best else "transparent"
                    val_str  = fmt.format(v)
                    best_star = " ⭐" if is_best else ""
                    if g is None:
                        gain_html = '<span style="color:#94A3B8;font-size:0.78rem">— baseline</span>'
                    elif abs(g) < 0.5:
                        gain_html = '<span style="color:#94A3B8;font-size:0.78rem">≈ 0%</span>'
                    else:
                        arrow    = "▲" if g > 0 else "▼"
                        gain_col = "#15803D" if g > 0 else "#DC2626"
                        gain_html = (
                            f'<span style="color:{gain_col};font-size:0.78rem;font-weight:700">'
                            f'{arrow} {abs(g):.0f}%</span>'
                        )
                    rows_html += (
                        f'<td style="background:{cell_bg}">'
                        f'<b>{val_str}{best_star}</b><br>{gain_html}</td>'
                    )
                rows_html += "</tr>"

            st.markdown(header_html + rows_html + "</tbody></table></div>",
                        unsafe_allow_html=True)

            # ── 3. % Gain grouped bar chart ──
            st.markdown("#### % Gain vs Random — All Metrics at a Glance")
            metric_labels_short = ["Reward", "Budget Met", "Calorie Met", "Both Goals", "Churn (↓better)"]
            non_baseline = [n for n in agent_names if n != baseline]

            fig_gain = go.Figure()
            for n in non_baseline:
                gains_list = []
                for metric_key, _, higher_better, _ in METRICS:
                    g = gain(metric_key, n, higher_better)
                    gains_list.append(g if g is not None else 0.0)
                fig_gain.add_trace(go.Bar(
                    name=n, x=metric_labels_short, y=gains_list,
                    marker_color=AGENT_COLORS.get(n, "#94A3B8"),
                    text=[f"{g:+.0f}%" for g in gains_list],
                    textposition="outside",
                ))

            fig_gain.add_hline(y=0, line_color="#94A3B8", line_width=1.5)
            fig_gain.update_layout(**PLOTLY_LAYOUT,
                barmode="group", height=360,
                xaxis=dict(**GRID),
                yaxis=dict(title="% Change vs Random Baseline",
                           **GRID, zeroline=True),
                margin=dict(t=50, b=20, l=60, r=20),
            )
            fig_gain.update_layout(legend=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#E2E8F0", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02,
            ))
            st.plotly_chart(fig_gain, use_container_width=True)

            # ── 4. Key Takeaways (high-contrast) ──
            st.markdown("#### Key Takeaways")
            takeaways = {
                "Random":     ("#94A3B8", "Baseline. Uniform random selection — no learning, no goal awareness."),
                "Greedy":     ("#E11D48", "Pure exploitation (ε=0): spams the highest-reward meal until fatigue collapses it. Classic exploration failure."),
                "ε-Greedy":   ("#F97316", "10% random exploration breaks the fatigue trap — small exploration budget, meaningful variety gain."),
                "Rule-Based": ("#EA580C", "Hand-tuned heuristic — preference + fatigue + budget score. No learning, but structured."),
                "LinUCB":     ("#CA8A04", "Contextual bandit — learns preferences but treats each meal as independent, so budget pacing fails."),
                "PPO":        ("#0F766E", "Best overall reward. Full MDP with temporal reasoning — learns to pace budget and calories across the week."),
                "PPO-L":      ("#7C3AED", "PPO-Lagrangian: explicit constraint enforcement via online dual updates on budget/calorie limits — trades some reward for compliance."),
            }
            for n in agent_names:
                col, text = takeaways.get(n, ("#94A3B8", n))
                rv = results[n]["mean_reward"]
                g  = gain("mean_reward", n)
                g_str = (f"+{g:.0f}% vs Random" if g and g > 0
                         else (f"{g:.0f}% vs Random" if g else "baseline"))
                st.markdown(
                    f'<div class="takeaway-row" style="border-left-color:{col}">'
                    f'  <div class="takeaway-head">'
                    f'    <span class="takeaway-name" style="color:{col}">{n}</span>'
                    f'    <span class="takeaway-meta">Reward {rv:+.1f} · {g_str}</span>'
                    f'  </div>'
                    f'  <div class="takeaway-body">{text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── TAB 3: Before vs After ──
        with tab3:
            st.markdown("### Before vs After Training")
            st.markdown("Left: **Random agent** — no learning. Right: **Trained PPO** — 800 episodes of experience.")

            col_b, col_a = st.columns(2)

            def render_episode_log(col, log, title, color):
                with col:
                    st.markdown(f"#### {title}")
                    total_r = sum(l["reward"] for l in log)
                    churns  = sum(1 for l in log if l["response"] == "churn")
                    accepted= sum(1 for l in log if "accept" in l["response"])
                    st.markdown(f"""
<div class="card">
<b>Total Reward:</b> <span style="color:{color};font-size:1.2em;font-weight:800">{total_r:+.1f}</span>
&nbsp;&nbsp; <b>Accepted:</b> {accepted}/{len(log)}
&nbsp;&nbsp; <b>Churns:</b> <span style="color:#DC2626">{churns}</span>
</div>
""", unsafe_allow_html=True)

                    prev_day = None
                    for step in log[:15]:
                        if step["day"] != prev_day:
                            st.markdown(f"**━━ {step['day']} ━━**")
                            prev_day = step["day"]
                        resp  = step["response"]
                        r_val = step.get("reward", 0)
                        r_col = "#15803D" if r_val >= 0 else "#DC2626"
                        if "accept" in resp:
                            badge_bg, badge_fg = "#ECFDF5", "#15803D"
                        elif resp == "reject":
                            badge_bg, badge_fg = "#FFF7ED", "#EA580C"
                        elif resp == "churn":
                            badge_bg, badge_fg = "#FEF2F2", "#DC2626"
                        else:
                            badge_bg, badge_fg = "#F1F5F9", "#475569"
                        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0;
            border-bottom:1px solid var(--border)">
  <span style="font-size:1.4em">{display_meal_emoji(step)}</span>
  <div style="flex:1;min-width:0">
    <b style="font-size:0.9em">{step['meal']}</b>
    <span style="font-size:0.75em;color:var(--text-mute);margin-left:6px">
      {step['category']} · ${step['price']}</span><br>
    <span style="font-size:0.72em;color:var(--text-mute)">{step['meal_time']}</span>
  </div>
  <span style="background:{badge_bg};color:{badge_fg};border-radius:6px;
               padding:2px 9px;font-size:.75rem;font-weight:700;white-space:nowrap">
    {RESPONSE_EMOJI.get(resp,'?')} {resp.upper()}</span>
  <b style="color:{r_col};min-width:42px;text-align:right">{r_val:+.1f}</b>
</div>
""", unsafe_allow_html=True)

            render_episode_log(col_b, log_before, "❌ Random Agent", "#DC2626")
            render_episode_log(col_a, log_after,  "✅ Trained PPO",  "#0F766E")

            st.markdown("#### Cumulative Reward Trajectory")
            fig = go.Figure()
            for log_data, name, color in [
                (log_before, "Random", "#DC2626"),
                (log_after,  "PPO",    "#0F766E"),
            ]:
                cumr = np.cumsum([l["reward"] for l in log_data])
                fig.add_trace(go.Scatter(
                    x=list(range(len(cumr))), y=cumr,
                    mode="lines+markers", name=name,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6),
                ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                xaxis=dict(title="Meal Step", **GRID),
                yaxis=dict(title="Cumulative Reward", **GRID),
                margin=dict(t=10, b=40, l=60, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── TAB 4: State & Reward Breakdown ──
        with tab4:
            st.markdown("### State Vector & Reward Decomposition")
            st.markdown("Inspect the trained PPO's internal reasoning on a single trained episode.")

            fat_data = [step["fatigue"] for step in log_after]
            fat_arr  = np.array(fat_data).T

            fig_fat = go.Figure(data=go.Heatmap(
                z=fat_arr,
                x=[f"Step {i+1}" for i in range(len(log_after))],
                y=CATEGORIES,
                colorscale="RdYlGn_r",
                colorbar=dict(title="Fatigue"),
                zmin=0, zmax=2,
            ))
            fig_fat.update_layout(**PLOTLY_LAYOUT, height=300,
                title="Category Fatigue Over Time (Trained PPO)",
                margin=dict(t=40, b=40, l=80, r=20),
            )
            st.plotly_chart(fig_fat, use_container_width=True)

            r_accept_vals  = [l.get("r_accept",  l["reward"]) for l in log_after]
            r_goal_vals    = [l.get("r_goal",    0.0)         for l in log_after]
            r_terminal_vals= [l.get("r_terminal",0.0)         for l in log_after]
            steps          = list(range(len(log_after)))

            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=steps, y=r_accept_vals,  name="r_accept",
                                   marker_color="#0F766E"))
            fig_r.add_trace(go.Bar(x=steps, y=r_goal_vals,    name="r_goal",
                                   marker_color="#15803D"))
            fig_r.add_trace(go.Bar(x=steps, y=r_terminal_vals,name="r_terminal",
                                   marker_color="#CA8A04"))
            fig_r.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=300,
                title="Reward Components Per Step (Trained PPO)",
                xaxis=dict(title="Meal Step", **GRID),
                yaxis=dict(title="Reward", **GRID),
                margin=dict(t=40, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_r, use_container_width=True)

            steps_all = list(range(len(log_after)))
            budgets   = [l["budget_spent"] for l in log_after]
            calories  = [l["calories_eaten"] for l in log_after]

            fig_g = make_subplots(rows=1, cols=2,
                subplot_titles=("Weekly Budget ($200)", "Weekly Calories (18,000 kcal)"))
            fig_g.add_trace(go.Scatter(x=steps_all, y=budgets, mode="lines+markers",
                line=dict(color="#EA580C"), name="Spent"), row=1, col=1)
            fig_g.add_hline(y=200, line_dash="dash", line_color="#DC2626",
                annotation_text="Budget limit", row=1, col=1)
            fig_g.add_trace(go.Scatter(x=steps_all, y=calories, mode="lines+markers",
                line=dict(color="#7C3AED"), name="Eaten"), row=1, col=2)
            fig_g.add_hline(y=18000, line_dash="dash", line_color="#DC2626",
                annotation_text="Cal limit", row=1, col=2)
            fig_g.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                margin=dict(t=40, b=40, l=60, r=20),
            )
            for i in range(1, 3):
                fig_g.update_xaxes(**GRID, row=1, col=i)
                fig_g.update_yaxes(**GRID, row=1, col=i)
            st.plotly_chart(fig_g, use_container_width=True)

        # ── TAB 5: PPO-Lagrangian Constraint Analysis ──
        with tab5:
            st.markdown("### ⚖️ PPO-Lagrangian (PPO-L): Primal-Dual Constrained RL")
            st.markdown("""
**Why PPO-L?** PPO folds budget and calorie goals into the reward as *soft bonuses* —
the agent can trade constraint satisfaction for higher acceptance rewards.
**PPO-Lagrangian** (Ray et al., 2019 — the standard safe-RL baseline) treats them
as **explicit constraints** via Lagrangian relaxation, with multipliers that adapt
online to enforce them, regardless of the reward landscape.
""")

            st.markdown("#### The Lagrangian Saddle-Point Problem")
            col_eq, col_exp = st.columns([1, 1])
            with col_eq:
                st.markdown("""
<div class="eq-card">
<b style="color:#7C3AED">Primal (policy maximises):</b><br>
L(π, λ) = E[Σ r_accept]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− λ_b · (C_b − 0.95)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− λ_c · (C_c − 0.95)<br><br>
<b style="color:#EA580C">Dual (multipliers maximise):</b><br>
λ_b ← λ_b + α · (C_b − 0.95)<br>
λ_c ← λ_c + α · (C_c − 0.95)<br><br>
<span style="color:#15803D">KKT condition at convergence:</span><br>
λ* · (C − d) = 0<br>
<span style="color:var(--text-mute)">↳ constraint satisfied OR multiplier=0</span>
</div>
""", unsafe_allow_html=True)
            with col_exp:
                st.markdown("""
<div class="card" style="line-height:1.8;font-size:.9rem">
<b style="color:#7C3AED">What this means in practice:</b><br><br>
🔺 If budget is <b>over</b> 95% → λ_b <b>increases</b><br>
&nbsp;&nbsp;&nbsp;→ expensive meals penalised harder<br>
&nbsp;&nbsp;&nbsp;→ policy shifts to cheaper options<br><br>
🔻 If budget is <b>under</b> 95% → λ_b <b>decreases</b><br>
&nbsp;&nbsp;&nbsp;→ penalty relaxes<br>
&nbsp;&nbsp;&nbsp;→ policy can optimise acceptance again<br><br>
⚖️ The multipliers are <b>self-tuning</b> — no manual<br>
&nbsp;&nbsp;&nbsp;reward engineering needed.<br><br>
<span style="color:#15803D">vs PPO: PPO uses fixed r_goal weights.<br>
PPO-L adapts its constraint pressure dynamically.</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("#### Lagrange Multiplier Evolution During Training")
            lambda_b = data.get("lambda_b_curve", [])
            lambda_c = data.get("lambda_c_curve", [])

            if lambda_b:
                eps_l = [x[0] for x in lambda_b]
                lb_v  = [x[1] for x in lambda_b]
                lc_v  = [x[1] for x in lambda_c]

                fig_lam = go.Figure()
                fig_lam.add_trace(go.Scatter(
                    x=eps_l, y=lb_v, mode="lines",
                    line=dict(color="#EA580C", width=2.5),
                    fill="tozeroy", fillcolor="rgba(234,88,12,0.10)",
                    name="λ_budget",
                ))
                fig_lam.add_trace(go.Scatter(
                    x=eps_l, y=lc_v, mode="lines",
                    line=dict(color="#15803D", width=2.5),
                    fill="tozeroy", fillcolor="rgba(21,128,61,0.08)",
                    name="λ_calorie",
                ))
                fig_lam.add_hline(y=0, line_dash="dot", line_color="#94A3B8",
                                  line_width=1)
                fig_lam.add_annotation(
                    x=eps_l[len(eps_l)//3], y=max(lb_v)*0.6,
                    text="Rising λ = agent learning<br>budget is tight → self-correcting",
                    showarrow=True, arrowcolor="#EA580C",
                    font=dict(color="#EA580C", size=10),
                    bgcolor="rgba(254,243,231,0.9)", bordercolor="#EA580C",
                )
                fig_lam.update_layout(**PLOTLY_LAYOUT, height=300,
                    xaxis=dict(title="Training Episode", **GRID),
                    yaxis=dict(title="Lagrange Multiplier Value", **GRID),
                    margin=dict(t=20, b=40, l=60, r=20),
                )
                st.plotly_chart(fig_lam, use_container_width=True)

            st.markdown("#### Constraint Attainment: All Agents")
            st.markdown("*Budget goal: spend ≤ 95% of $200/week. Calorie goal: eat ≤ 95% of 18,000 kcal/week.*")

            agent_names = list(results.keys())
            budget_pcts  = [results[n]["budget_pct"]       for n in agent_names]
            calorie_pcts = [results[n]["calorie_pct"]      for n in agent_names]
            both_pcts    = [results[n]["constraint_ok_pct"] for n in agent_names]
            colors       = [AGENT_COLORS.get(n, "#94A3B8") for n in agent_names]

            fig_con = make_subplots(rows=1, cols=3,
                subplot_titles=(
                    "Budget Goal Met (%)",
                    "Calorie Goal Met (%)",
                    "BOTH Goals Met (%) ← key metric",
                ))
            for col_i, vals in enumerate([budget_pcts, calorie_pcts, both_pcts], 1):
                fig_con.add_trace(go.Bar(
                    x=agent_names, y=vals,
                    marker_color=colors,
                    text=[f"{v:.0f}%" for v in vals],
                    textposition="outside",
                    showlegend=False,
                ), row=1, col=col_i)
                fig_con.add_hline(y=80, line_dash="dash",
                                  line_color="#DC2626", line_width=1,
                                  annotation_text="80% target",
                                  annotation_font_color="#DC2626",
                                  row=1, col=col_i)

            fig_con.update_layout(**PLOTLY_LAYOUT, height=360,
                margin=dict(t=50, b=20, l=40, r=20))
            for c in range(1, 4):
                fig_con.update_xaxes(**GRID, row=1, col=c)
                fig_con.update_yaxes(**GRID, range=[0, 115], row=1, col=c)
            st.plotly_chart(fig_con, use_container_width=True)

            st.markdown("#### Episode Comparison: PPO vs PPO-L")
            st.markdown("Same user, same week. PPO maximises reward but may overspend. "
                        "PPO-L trades some reward for tighter constraint compliance.")

            log_ppo_d  = data.get("log_after", [])
            log_ppol_d = data.get("log_ppol", data.get("log_cpo", []))

            if log_ppo_d and log_ppol_d:
                col_p, col_c = st.columns(2)
                for col, log_d, label, color in [
                    (col_p, log_ppo_d,  "PPO",   "#0F766E"),
                    (col_c, log_ppol_d, "PPO-L", "#7C3AED"),
                ]:
                    with col:
                        total_r = sum(l["reward"] for l in log_d)
                        b_spent = log_d[-1]["budget_spent"] if log_d else 0
                        c_eaten = log_d[-1]["calories_eaten"] if log_d else 0
                        b_ok = "✅" if b_spent  <= 200*0.95 else "❌"
                        c_ok = "✅" if c_eaten <= 18000*0.95 else "❌"
                        st.markdown(f"""
<div class="card">
<b style="color:{color};font-size:1.05em">{label}</b><br>
Reward: <b>{total_r:+.1f}</b><br>
Budget: ${b_spent:.0f} / $200 {b_ok}<br>
Calories: {c_eaten:,} / 18,000 {c_ok}
</div>
""", unsafe_allow_html=True)
                        prev_day = None
                        for step in log_d[:12]:
                            if step["day"] != prev_day:
                                st.markdown(f"**━━ {step['day']} ━━**")
                                prev_day = step["day"]
                            resp  = step["response"]
                            r_val = step.get("reward", 0)
                            r_col = "#15803D" if r_val >= 0 else "#DC2626"
                            st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;padding:4px 0;
            border-bottom:1px solid var(--border)">
  <span style="font-size:1.2em">{display_meal_emoji(step)}</span>
  <div style="flex:1;min-width:0">
    <b style="font-size:0.85em">{step['meal']}</b>
    <span style="font-size:0.7em;color:var(--text-mute);margin-left:4px">
      ${step['price']} · {step['calories']}kcal</span>
  </div>
  <span style="font-size:0.85em">{RESPONSE_EMOJI.get(resp,'?')}</span>
  <b style="color:{r_col};font-size:0.85em">{r_val:+.1f}</b>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🎲 Exploration vs Exploitation: Greedy vs ε-Greedy")
            st.markdown(
                "Greedy and ε-Greedy use the same Q-update rule — the only difference is "
                "ε=0 vs ε=0.1. This isolates the pure value of exploration."
            )

            col_g, col_eg = st.columns(2)
            with col_g:
                st.markdown("""
<div class="card">
<b style="color:#E11D48;font-size:1.0em">Greedy (ε=0) — Pure Exploitation</b><br><br>
<span style="font-size:0.88em;line-height:1.7">
• Discovers Japanese food → high reward<br>
• Spams Japanese every meal<br>
• Fatigue[Japanese] climbs to 2.0+<br>
• User starts rejecting → reward collapses<br>
• Stuck: Q[Japanese] is still highest<br>
&nbsp;&nbsp;so it never tries alternatives<br><br>
<b style="color:#DC2626">Result: great early, terrible long-term</b>
</span>
</div>
""", unsafe_allow_html=True)

            with col_eg:
                st.markdown("""
<div class="card">
<b style="color:#EA580C;font-size:1.0em">ε-Greedy (ε=0.1) — Guided Exploration</b><br><br>
<span style="font-size:0.88em;line-height:1.7">
• 90% of the time: picks best known meal<br>
• 10% of the time: tries a random meal<br>
• Discovers that Light/Korean also acceptable<br>
• Fatigue stays low across categories<br>
• Q-values stay up to date across the menu<br><br>
<b style="color:#15803D">Result: consistently good, avoids ruts</b>
</span>
</div>
""", unsafe_allow_html=True)

            if "Greedy" in results and "ε-Greedy" in results:
                greedy_names = [n for n in ["Random", "Greedy", "ε-Greedy"] if n in results]
                g_metrics = ["mean_reward", "budget_pct", "constraint_ok_pct"]
                g_labels  = ["Reward", "Budget Met %", "Both Goals Met %"]

                fig_gex = make_subplots(rows=1, cols=3, subplot_titles=g_labels)
                for ci, (mk, ml) in enumerate(zip(g_metrics, g_labels), 1):
                    fig_gex.add_trace(go.Bar(
                        x=greedy_names,
                        y=[results[n][mk] for n in greedy_names],
                        marker_color=[AGENT_COLORS.get(n, "#94A3B8") for n in greedy_names],
                        text=[f"{results[n][mk]:+.1f}" if mk == "mean_reward"
                              else f"{results[n][mk]:.0f}%" for n in greedy_names],
                        textposition="outside",
                        showlegend=False,
                    ), row=1, col=ci)
                fig_gex.update_layout(**PLOTLY_LAYOUT, height=280,
                    margin=dict(t=40, b=20, l=40, r=20))
                for ci in range(1, 4):
                    fig_gex.update_xaxes(**GRID, row=1, col=ci)
                    fig_gex.update_yaxes(**GRID, row=1, col=ci)
                st.plotly_chart(fig_gex, use_container_width=True)

                g_r  = results.get("Greedy",   {}).get("mean_reward", 0)
                eg_r = results.get("ε-Greedy", {}).get("mean_reward", 0)
                if abs(g_r) > 1e-3:
                    exploration_lift = (eg_r - g_r) / abs(g_r) * 100
                    st.markdown(f"""
<div style="padding:10px 16px;border-left:3px solid #EA580C;
background:#FFF7ED;border-radius:0 8px 8px 0;margin:8px 0;color:var(--text)">
<b style="color:#EA580C">Exploration lift:</b>
Adding ε=0.1 exploration improves reward over pure Greedy by
<b style="color:#0F766E">{exploration_lift:+.1f}%</b>.
That's the value of 1-in-10 random meal decisions.
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
<div class="meal-card" style="margin-top:8px">
<b style="color:#7C3AED;font-size:1.05em">📐 The framing contribution</b><br><br>
Most food recommendation papers fold goals into PPO/DQN reward shaping.
TasteFlow casts the same problem as a <b>constrained MDP</b> and solves it with
<b>PPO-Lagrangian</b> (Ray et al., 2019) — budget and calorie targets become
explicit constraints with online dual updates, not hand-tuned reward weights.
<br><br>
The KKT conditions imply that at convergence, either the constraint is satisfied
or the multiplier is zero (it wasn't binding). In practice, convergence depends
on learning rates and training length — we report empirical attainment in the
results table, not formal guarantees.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MODE 2: BE THE USER  (Phone mockup + Analyst panel)
# ═══════════════════════════════════════════════════════════════
elif mode == "🎮 Be the User":

    st.markdown("## 🎮 Be the User")
    st.markdown(
        '<p class="page-sub">Live demo: the RL agent recommends meals on your phone — '
        'you respond like a real user, and the analyst panel reveals what the agent is thinking.</p>',
        unsafe_allow_html=True,
    )

    data = load_results()
    if data is None:
        st.warning("⚠️ No trained model found. Run `python train.py` first.")
        st.stop()

    # ── Agent selector (full width, above the two-pane layout) ──
    agent_choice = st.radio(
        "Recommender agent",
        ["🤖 PPO (maximises reward)",
         "⚖️ PPO-L (enforces budget & calorie constraints)",
         "🎯 Greedy (pure exploitation — ε=0)",
         "🎲 ε-Greedy (10% exploration)"],
        horizontal=True,
    )
    use_ppol    = "PPO-L"    in agent_choice
    use_greedy  = "Greedy"   in agent_choice and "ε" not in agent_choice
    use_egreedy = "ε-Greedy" in agent_choice

    with st.expander("🍽️ Personalize this demo user", expanded=True):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            preferred_cat = st.selectbox(
                "Favorite cuisine",
                CATEGORIES,
                index=CATEGORIES.index("Japanese"),
            )
            intent = st.selectbox(
                "Today's intent",
                ["Balanced week", "Save money", "Eat healthier",
                 "Quick meal", "High protein", "Comfort food"],
            )
        with pcol2:
            weekly_budget = st.slider("Weekly budget", 120, 320, 200, step=10)
            max_price = st.slider("Max price per meal", 8, 30, 25, step=1)
        with pcol3:
            weekly_calories = st.slider("Weekly calories", 12000, 24000, 18000, step=1000)
            max_calories_per_meal = st.slider("Max calories per meal", 450, 1200, 1050, step=50)

        preferred_tags = st.multiselect(
            "Taste or lifestyle tags you like",
            ALL_MEAL_TAGS,
            default=["healthy", "comfort"],
        )
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            avoided_tags = st.multiselect(
                "Hard filters: avoid tags",
                ALL_MEAL_TAGS,
                default=[],
            )
        with fcol2:
            disliked_cats = st.multiselect(
                "Hard filters: avoid cuisines",
                [c for c in CATEGORIES if c != preferred_cat],
                default=[],
            )

        profile_config = {
            "preferred_cat": preferred_cat,
            "intent": intent,
            "weekly_budget": weekly_budget,
            "weekly_calories": weekly_calories,
            "max_price": max_price,
            "max_calories_per_meal": max_calories_per_meal,
            "preferred_tags": tuple(sorted(preferred_tags)),
            "avoided_tags": tuple(sorted(avoided_tags)),
            "disliked_cats": tuple(sorted(disliked_cats)),
        }
        st.caption(
            "These choices become the initial MDP state and action mask; "
            "the recommender still selects from its learned policy."
        )

    # ── Initialize / reset session ──
    reset_needed = (
        "game_env" not in st.session_state
        or st.session_state.get("game_reset")
        or st.session_state.get("game_agent_type") != agent_choice
        or st.session_state.get("game_profile_config") != profile_config
    )
    if reset_needed:
        profile = build_user_profile(profile_config)
        st.session_state.game_env        = TasteFlowEnv(
            profile_config["weekly_budget"],
            profile_config["weekly_calories"],
            profile,
        )
        st.session_state.game_state      = st.session_state.game_env.reset()
        st.session_state.game_log        = []
        st.session_state.cumulative_r    = 0.0
        st.session_state.game_done       = False
        st.session_state.game_reset      = False
        st.session_state.game_agent_type = agent_choice
        st.session_state.game_profile_config = profile_config
        st.session_state.ppo_agent       = make_trained_ppo(data["ppo_weights"])
        st.session_state.greedy_agent    = GreedyAgent()
        st.session_state.egreedy_agent   = EpsilonGreedyAgent(epsilon=0.1)
        # Backward-compat: load 'ppol_weights' (new) or 'cpo_weights' (legacy pkl)
        ppol_w = data.get("ppol_weights", data.get("cpo_weights"))
        if ppol_w is not None:
            st.session_state.ppol_agent  = make_trained_ppol(ppol_w)
        st.session_state.current_rec     = None
        st.session_state.current_probs   = None

    env = st.session_state.game_env
    if use_greedy:
        agent = st.session_state.greedy_agent
    elif use_egreedy:
        agent = st.session_state.egreedy_agent
    elif use_ppol and "ppol_agent" in st.session_state:
        agent = st.session_state.ppol_agent
    else:
        agent = st.session_state.ppo_agent
    s        = st.session_state.game_state
    game_log = st.session_state.game_log

    # ── Compute current step + recommendation (only if not done) ──
    step_idx   = env.step_idx
    day_idx    = min(step_idx // len(MEAL_TIMES), 6)
    mt_idx     = step_idx % len(MEAL_TIMES)
    day_label  = DAYS[day_idx] if day_idx < 7 else "Done"
    meal_label = MEAL_TIMES[mt_idx] if not st.session_state.game_done else "—"

    if not st.session_state.game_done:
        valid  = env.get_valid_actions()
        probs  = get_policy_probs(agent, s, valid)
        action = valid[int(np.argmax([probs[i] for i in valid]))]
        meal   = MEALS[action]
        cat_idx    = CATEGORIES.index(meal["category"])
        fatigue_v  = env.fatigue[cat_idx]
        budget_rem = env.weekly_budget - env.budget_spent
        cal_rem    = env.weekly_calories - env.calories_eaten
        rec_reason = explain_recommendation(meal, env, probs[action])
    else:
        meal = None
        fatigue_v = 0.0
        rec_reason = ""

    # ── Two-column layout: phone left, analyst right ──
    phone_col, analyst_col = st.columns([1, 2], gap="large")

    # ╔══════════════════════════ PHONE COL ══════════════════════════╗
    with phone_col:
        # Marker that triggers the phone-frame CSS via :has()
        st.markdown('<div class="phone-marker"></div>', unsafe_allow_html=True)

        if st.session_state.game_done:
            # ── In-app weekly report (replaces meal card when episode ends) ──
            summ = env.summary()
            b_ok = "✅ Met" if summ["budget_ok"] else "❌ Missed"
            c_ok = "✅ Met" if summ["calorie_ok"] else "❌ Missed"
            st.markdown(f"""
<div class="phone-statusbar">
  <span>📅 <b>Week complete</b></span>
  <span>⭐ <b style="color:var(--primary)">{summ['total_reward']:+.1f}</b></span>
</div>
<div class="phone-meal" style="text-align:center">
  <div style="font-size:2.5em">🏁</div>
  <div style="font-weight:800;font-size:1.1rem;margin-top:4px">Your week summary</div>
  <div style="color:var(--text-mute);font-size:.78rem;margin-top:2px">
    Tap "Play Again" on the right to restart
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
  <div class="card" style="padding:10px 12px">
    <div class="takeaway-meta">💰 Budget</div>
    <div style="font-weight:700">{b_ok}</div>
    <div class="takeaway-meta">${summ['budget_spent']:.0f} spent</div>
  </div>
  <div class="card" style="padding:10px 12px">
    <div class="takeaway-meta">🔥 Calories</div>
    <div style="font-weight:700">{c_ok}</div>
    <div class="takeaway-meta">{summ['calories_eaten']:,} kcal</div>
  </div>
  <div class="card" style="padding:10px 12px">
    <div class="takeaway-meta">💨 Churns</div>
    <div style="font-weight:700">{summ['churns']}</div>
  </div>
  <div class="card" style="padding:10px 12px">
    <div class="takeaway-meta">⭐ Reward</div>
    <div style="font-weight:700;color:var(--primary)">{summ['total_reward']:+.1f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        else:
            # ── Status bar: Day · Meal · Reward (inline, no truncation) ──
            budget_left = max(env.weekly_budget - env.budget_spent, 0)
            cal_left    = max(env.weekly_calories - env.calories_eaten, 0)
            budget_pct  = min(env.budget_spent / env.weekly_budget * 100, 100)
            cal_pct     = min(env.calories_eaten / env.weekly_calories * 100, 100)

            st.markdown(f"""
<div class="phone-statusbar">
  <span>📅 <b>{day_label}</b></span>
  <span>🍽️ <b>{meal_label}</b></span>
  <span>⭐ <b style="color:var(--primary)">{st.session_state.cumulative_r:+.1f}</b></span>
</div>
<div class="phone-progress">
  <div>💰 <b>${budget_left:.0f}</b> <span>/ ${env.weekly_budget:.0f}</span>
       <div class="bar"><span style="width:{budget_pct:.0f}%;background:var(--primary)"></span></div></div>
  <div>🔥 <b>{cal_left:,.0f}</b> <span>/ {env.weekly_calories:,.0f}</span>
       <div class="bar"><span style="width:{cal_pct:.0f}%;background:#7C3AED"></span></div></div>
</div>
""", unsafe_allow_html=True)

            # ── Recommendation card (mobile-friendly, 360px width) ──
            fa_class = "pill-fa-hi" if fatigue_v > 1.0 else "pill-fa-ok"
            st.markdown(f"""
<div class="phone-meal">
  <div style="font-size:.7rem;color:var(--text-mute);margin-bottom:6px">
    🤖 Recommended for you
  </div>
  <div class="meal-main" style="display:flex;align-items:center;gap:10px">
    <span class="meal-emoji" style="font-size:2.6em;line-height:1">{meal['emoji']}</span>
    <div style="flex:1;min-width:0">
      <div class="meal-name" style="font-weight:800;font-size:1rem;color:var(--text)">{meal['name']}</div>
      <div style="font-size:.72rem;color:var(--text-mute)">{meal['category']}</div>
    </div>
  </div>
  <div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap">
    <span class="pill pill-money">💰 ${meal['price']}</span>
    <span class="pill pill-cal">🔥 {meal['calories']} kcal</span>
    <span class="pill {fa_class}">😮‍💨 {fatigue_v:.1f}</span>
  </div>
  <div style="font-size:.72rem;color:var(--text-mute);line-height:1.35;margin-top:10px">
    Why this: {rec_reason}
  </div>
</div>
""", unsafe_allow_html=True)

            # ── 2x2 semantic-color buttons ──
            clicked = None
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.markdown('<div class="btn-accept">', unsafe_allow_html=True)
                if st.button("✅ Accept", use_container_width=True, key="b_accept"):
                    clicked = "accept_override"
                st.markdown('</div>', unsafe_allow_html=True)
            with bcol2:
                st.markdown('<div class="btn-browse">', unsafe_allow_html=True)
                if st.button("👍 After browse", use_container_width=True, key="b_browse"):
                    clicked = "browse_override"
                st.markdown('</div>', unsafe_allow_html=True)

            bcol3, bcol4 = st.columns(2)
            with bcol3:
                st.markdown('<div class="btn-reject">', unsafe_allow_html=True)
                if st.button("😐 Reject", use_container_width=True, key="b_reject"):
                    clicked = "reject_override"
                st.markdown('</div>', unsafe_allow_html=True)
            with bcol4:
                st.markdown('<div class="btn-churn">', unsafe_allow_html=True)
                if st.button("💨 Churn", use_container_width=True, key="b_churn"):
                    clicked = "churn_override"
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Recent 3 history (order-stream feel) ──
            if game_log:
                rows_html = ""
                for entry in reversed(game_log[-3:]):
                    resp  = entry.get("response", "")
                    r_val = entry.get("reward", 0)
                    color = "var(--accept)" if r_val >= 0 else "var(--churn)"
                    rows_html += f"""
<div class="phone-history-row">
  <span style="font-size:1.1em">{display_meal_emoji(entry)}</span>
  <span class="name"><b>{entry['meal']}</b></span>
  <span style="color:var(--text-mute);font-size:.7rem">{RESPONSE_EMOJI.get(resp,'?')}</span>
  <b style="color:{color}">{r_val:+.1f}</b>
</div>"""
                st.markdown(
                    f'<div class="phone-history">'
                    f'<div style="font-size:.72rem;color:var(--text-mute);margin-bottom:4px">'
                    f'Recent orders</div>{rows_html}</div>',
                    unsafe_allow_html=True,
                )

            # ── Apply click outside the markdown so analyst can rerun ──
            if clicked:
                orig_respond = env.user_profile.respond

                def forced_respond(meal, *args, _clicked=clicked, **kwargs):
                    r_map = {
                        "accept_override": ("accept", 1.5),
                        "browse_override": ("accept_browse", 0.8),
                        "reject_override": ("reject", -0.1),
                        "churn_override":  ("churn", -0.8),
                    }
                    return r_map[_clicked]

                env.user_profile.respond = forced_respond
                next_s, reward, done, info = env.step(action)
                env.user_profile.respond  = orig_respond

                if use_greedy or use_egreedy:
                    agent.update(action, s, reward)

                st.session_state.game_state    = next_s
                st.session_state.cumulative_r += reward
                st.session_state.game_done     = done
                game_log.append(info)
                st.rerun()

    # ╔════════════════════════ ANALYST COL ════════════════════════╗
    with analyst_col:
        # ── Agent banner (context for the chosen agent) ──
        if use_ppol:
            st.markdown("""
<div class="mode-banner" style="border-color:#DDD6FE;background:linear-gradient(135deg,#F5F3FF,#FFFFFF)">
⚖️ <b style="color:#7C3AED">PPO-Lagrangian Mode:</b> This agent's policy is penalised when budget or
calorie spending runs ahead of pace — Lagrange multipliers auto-tune the constraint pressure.
Watch how it picks lighter/cheaper meals as the week progresses.
</div>
""", unsafe_allow_html=True)
        elif use_greedy:
            st.markdown("""
<div class="mode-banner" style="border-color:#FECDD3;background:linear-gradient(135deg,#FFF1F2,#FFFFFF)">
🎯 <b style="color:#E11D48">Greedy Mode (ε=0):</b> Pure exploitation — always picks the
meal with the highest average reward seen so far. Watch it lock onto your preferred category
and spam it until fatigue causes rejections.
</div>
""", unsafe_allow_html=True)
        elif use_egreedy:
            st.markdown("""
<div class="mode-banner" style="border-color:#FED7AA;background:linear-gradient(135deg,#FFF7ED,#FFFFFF)">
🎲 <b style="color:#EA580C">ε-Greedy Mode (ε=0.1):</b> 90% exploitation, 10% random exploration.
Notice how the occasional random pick keeps variety up and prevents the fatigue trap
that pure Greedy falls into.
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="mode-banner">
🤖 <b style="color:#0F766E">PPO Mode:</b> Trained on 800 episodes. Maximises full reward
including soft goal bonuses. Best at long-horizon pacing.
</div>
""", unsafe_allow_html=True)

        if st.session_state.game_done:
            # ── End-of-week analyst view: pie chart + reset ──
            cats_chosen = [l["category"] for l in game_log if "accept" in l.get("response","")]
            if cats_chosen:
                from collections import Counter
                cat_counts = Counter(cats_chosen)
                fig = go.Figure(go.Pie(
                    labels=list(cat_counts.keys()),
                    values=list(cat_counts.values()),
                    hole=0.5,
                    marker_colors=["#0F766E","#EA580C","#7C3AED","#CA8A04",
                                   "#15803D","#DC2626","#06B6D4"],
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=300,
                    title="What you ate this week",
                    margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

            if st.button("🔄 Play Again", use_container_width=True):
                st.session_state.game_reset = True
                st.rerun()

        else:
            # ── Why did the agent pick this? ──
            with st.expander("🧠 Agent reasoning — top-3 probabilities + state pace",
                             expanded=True):
                st.markdown(
                    f"**Personalized state:** prefers **{env.user_profile.preferred_cat}**, "
                    f"intent **{profile_config['intent']}**, "
                    f"liked tags **{', '.join(profile_config['preferred_tags']) or 'none'}**."
                )
                top3_idx = sorted(valid, key=lambda i: -probs[i])[:3]
                for rank, idx in enumerate(top3_idx):
                    m = MEALS[idx]
                    pct = probs[idx] * 100
                    bar_w = int(pct * 3)
                    why = explain_recommendation(m, env, probs[idx])
                    st.markdown(f"""
<div style="padding:7px 0;border-bottom:1px solid var(--border)">
  <div style="display:flex;align-items:center;gap:10px">
    <span style="width:20px;color:var(--text-mute)">{rank+1}.</span>
    <span style="font-size:1.2em">{m['emoji']}</span>
    <span style="flex:1;font-size:0.9em;color:var(--text)"><b>{m['name']}</b></span>
    <div style="background:#E2E8F0;border-radius:4px;width:140px;height:10px;overflow:hidden">
      <div style="background:var(--primary);width:{min(bar_w,140)}px;height:10px"></div>
    </div>
    <span style="font-size:0.85em;color:var(--primary);width:50px;text-align:right;font-weight:700">
      {pct:.1f}%</span>
  </div>
  <div style="font-size:.74rem;color:var(--text-mute);line-height:1.35;margin-left:30px">
    {why}
  </div>
</div>
""", unsafe_allow_html=True)

                st.markdown("")
                budget_pace = budget_rem / max(21 - step_idx, 1)
                cal_pace    = cal_rem    / max(21 - step_idx, 1)
                st.markdown(f"""
**Budget pace:** ${budget_pace:.1f}/meal remaining → meal costs **${meal['price']}**
{"✅ Within budget pace" if meal['price'] <= budget_pace else "⚠️ Over budget pace"}

**Calorie pace:** {cal_pace:.0f} kcal/meal remaining → meal is **{meal['calories']} kcal**
{"✅ Within calorie pace" if meal['calories'] <= cal_pace else "⚠️ Over calorie pace"}

**Fatigue[{meal['category']}]:** {fatigue_v:.2f}
{"⚠️ High fatigue — agent is taking a risk" if fatigue_v > 1.0 else "✅ Low fatigue — good variety choice"}
""")

            # ── Live fatigue chart ──
            st.markdown("##### 😮‍💨 Live fatigue by category")
            fat_vals = list(env.fatigue)
            fig_radar = go.Figure(go.Bar(
                x=CATEGORIES, y=fat_vals,
                marker_color=["#DC2626" if v > 1.0 else
                              "#EA580C" if v > 0.5 else "#0F766E"
                              for v in fat_vals],
                text=[f"{v:.1f}" for v in fat_vals],
                textposition="outside",
            ))
            fig_radar.update_layout(**PLOTLY_LAYOUT, height=240,
                xaxis=dict(**GRID),
                yaxis=dict(**GRID, range=[0, 3]),
                margin=dict(t=10, b=30, l=30, r=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # ── Full meal history ──
            if game_log:
                st.markdown("##### 📋 Full meal history")
                for entry in reversed(game_log):
                    resp  = entry.get("response","")
                    r_val = entry.get("reward", 0)
                    r_col = "#15803D" if r_val >= 0 else "#DC2626"
                    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0;
            border-bottom:1px solid var(--border)">
  <span style="font-size:1.2em">{display_meal_emoji(entry)}</span>
  <span style="flex:1;min-width:0;font-size:0.85em">
    <span style="color:var(--text-mute)">{entry['day']} {entry['meal_time']}</span>
    — <b>{entry['meal']}</b>
  </span>
  <span style="font-size:0.8em;color:var(--text-mute)">
    {RESPONSE_EMOJI.get(resp,'?')} {resp}</span>
  <b style="color:{r_col};font-size:0.9em">{r_val:+.1f}</b>
</div>
""", unsafe_allow_html=True)

            # ── Reset ──
            st.divider()
            if st.button("🔄 Reset Episode", use_container_width=True):
                st.session_state.game_reset = True
                st.rerun()
