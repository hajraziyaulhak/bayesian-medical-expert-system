import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bayesian Medical Expert System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background-color: #f4f7fb;
    color: #172033;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Make normal text dark and readable */
p, li, label, span, div {
    color: #172033;
}

/* ---------- HERO ---------- */

.hero {
    background: linear-gradient(135deg, #102a43, #1d4e89);
    padding: 32px 36px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(16, 42, 67, 0.18);
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    font-weight: 750;
    margin: 0;
}

.hero p {
    color: #dbeafe !important;
    font-size: 17px;
    margin-top: 8px;
}

/* ---------- SECTION HEADINGS ---------- */

.section-title {
    color: #102a43 !important;
    font-size: 25px;
    font-weight: 750;
    margin-top: 28px;
    margin-bottom: 16px;
}

/* ---------- CARDS ---------- */

.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #dbe3ed;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    margin-bottom: 18px;
}

.card h2,
.card h3 {
    color: #102a43 !important;
}

/* ---------- METRIC CARDS ---------- */

.metric-card {
    background: white;
    padding: 20px 14px;
    border-radius: 18px;
    border: 1px solid #dbe3ed;
    text-align: center;
    min-height: 125px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
}

.metric-title {
    color: #526579 !important;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #102a43 !important;
    font-size: 29px;
    font-weight: 800;
    margin-top: 7px;
}

/* ---------- RESULT CARD ---------- */

.result-card {
    background: linear-gradient(135deg, #ffffff, #eef6ff);
    border: 1px solid #c9dff5;
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(29, 78, 137, 0.10);
}

.result-label {
    color: #526579 !important;
    font-size: 16px;
    font-weight: 600;
}

.result-percentage {
    color: #155e9e !important;
    font-size: 58px;
    font-weight: 850;
    line-height: 1.1;
    margin: 8px 0;
}

.result-subtitle {
    color: #526579 !important;
    font-size: 14px;
}

/* ---------- INFO BOX ---------- */

.info-box {
    background: #eef6ff;
    border-left: 5px solid #2673c8;
    padding: 18px 20px;
    border-radius: 12px;
    margin: 15px 0;
}

.info-box b {
    color: #123b60 !important;
}

/* ---------- DECISION ---------- */

.decision-card {
    background: white;
    border: 1px solid #cbd8e6;
    padding: 23px;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
}

.decision-card h3 {
    color: #102a43 !important;
    margin-top: 0;
}

.decision-card p {
    color: #334e68 !important;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background-color: #102a43;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* ---------- SELECTBOX ---------- */

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff;
    color: #172033;
}

/* ---------- BUTTON ---------- */

.stButton > button {
    background: #1d6fb8;
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    padding: 0.65rem 1rem;
}

.stButton > button:hover {
    background: #155e9e;
    color: white;
}

/* ---------- TABLES ---------- */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- EXPANDERS ---------- */

.streamlit-expanderHeader {
    color: #102a43 !important;
    font-weight: 700;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #687b8f !important;
    font-size: 13px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🧠 Bayesian Medical Expert System</h1>

<p>
AI-powered probabilistic decision support using a
Bayesian Belief Network
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# BAYESIAN NETWORK
# ============================================================

model = DiscreteBayesianNetwork([
    ("Flu", "Fever"),
    ("Flu", "Cough"),
    ("Flu", "Fatigue")
])


# ============================================================
# PRIOR PROBABILITY
# ============================================================

cpd_flu = TabularCPD(
    variable="Flu",
    variable_card=2,
    values=[
        [0.70],
        [0.30]
    ],
    state_names={
        "Flu": ["No", "Yes"]
    }
)


# ============================================================
# FEVER CPT
# ============================================================

cpd_fever = TabularCPD(
    variable="Fever",
    variable_card=2,
    values=[
        [0.80, 0.20],
        [0.20, 0.80]
    ],
    evidence=["Flu"],
    evidence_card=[2],
    state_names={
        "Fever": ["No", "Yes"],
        "Flu": ["No", "Yes"]
    }
)


# ============================================================
# COUGH CPT
# ============================================================

cpd_cough = TabularCPD(
    variable="Cough",
    variable_card=2,
    values=[
        [0.75, 0.25],
        [0.25, 0.75]
    ],
    evidence=["Flu"],
    evidence_card=[2],
    state_names={
        "Cough": ["No", "Yes"],
        "Flu": ["No", "Yes"]
    }
)


# ============================================================
# FATIGUE CPT
# ============================================================

cpd_fatigue = TabularCPD(
    variable="Fatigue",
    variable_card=2,
    values=[
        [0.70, 0.30],
        [0.30, 0.70]
    ],
    evidence=["Flu"],
    evidence_card=[2],
    state_names={
        "Fatigue": ["No", "Yes"],
        "Flu": ["No", "Yes"]
    }
)


model.add_cpds(
    cpd_flu,
    cpd_fever,
    cpd_cough,
    cpd_fatigue
)

model.check_model()

inference = VariableElimination(model)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🩺 Patient Assessment")

st.sidebar.write(
    "Provide the available patient evidence."
)

fever = st.sidebar.selectbox(
    "🌡️ Fever",
    ["Not provided", "Yes", "No"]
)

cough = st.sidebar.selectbox(
    "😷 Cough",
    ["Not provided", "Yes", "No"]
)

fatigue = st.sidebar.selectbox(
    "😴 Fatigue",
    ["Not provided", "Yes", "No"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🧠 Bayesian Model")

st.sidebar.write("Disease variable: **Flu**")
st.sidebar.write("Evidence variables: **3**")
st.sidebar.write("Inference engine: **Variable Elimination**")

run_diagnosis = st.sidebar.button(
    "🔍 Perform Bayesian Diagnosis",
    use_container_width=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "diagnosed" not in st.session_state:
    st.session_state.diagnosed = False

if run_diagnosis:
    st.session_state.diagnosed = True


# ============================================================
# BUILD EVIDENCE
# ============================================================

evidence = {}

if fever != "Not provided":
    evidence["Fever"] = fever

if cough != "Not provided":
    evidence["Cough"] = cough

if fatigue != "Not provided":
    evidence["Fatigue"] = fatigue


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.diagnosed:

    st.markdown("""
    <div class="card">

    <h2>Welcome to the Decision Support System</h2>

    <p>
    This system estimates the probability of Flu using a
    <b>Bayesian Belief Network</b>.
    </p>

    <div class="info-box">

    <b>Probabilistic reasoning pipeline</b><br><br>

    Patient Evidence
    → Bayesian Network
    → Probabilistic Inference
    → Posterior Probability
    → Decision Support

    </div>

    <p>
    Unlike a simple rule-based system, the result changes
    according to the probability of each observed symptom.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>📌 Model Overview</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Prior Flu Probability</div>
        <div class="metric-value">30%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Evidence Variables</div>
        <div class="metric-value">3</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Reasoning Method</div>
        <div class="metric-value">BBN</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>🕸️ Bayesian Network</div>",
        unsafe_allow_html=True
    )


# ============================================================
# DIAGNOSIS
# ============================================================

else:

    # --------------------------------------------------------
    # BAYESIAN INFERENCE
    # --------------------------------------------------------

    result = inference.query(
        variables=["Flu"],
        evidence=evidence if evidence else None
    )

    flu_index = result.state_names["Flu"].index("Yes")

    flu_probability = float(
        result.values[flu_index]
    )

    no_flu_probability = 1 - flu_probability

    prior_probability = 0.30

    probability_change = (
        flu_probability - prior_probability
    ) * 100


    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    if flu_probability >= 0.75:
        uncertainty = "Low uncertainty"
        interpretation = "The observed evidence strongly supports Flu."

    elif flu_probability >= 0.50:
        uncertainty = "Moderate uncertainty"
        interpretation = "The evidence favors Flu, but the result is not conclusive."

    elif flu_probability >= 0.25:
        uncertainty = "Moderate uncertainty"
        interpretation = "The evidence does not strongly favor either outcome."

    else:
        uncertainty = "Low uncertainty"
        interpretation = "The evidence favors No Flu."


    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>📊 Bayesian Diagnosis Result</div>",
        unsafe_allow_html=True
    )

    # Large percentage + result
    result_col, chart_col = st.columns([1, 1.5])

    with result_col:

        st.markdown(f"""
        <div class="result-card">

        <div class="result-label">
        Posterior Probability of Flu
        </div>

        <div class="result-percentage">
        {flu_probability * 100:.2f}%
        </div>

        <div class="result-subtitle">
        Calculated using Bayesian probabilistic inference
        </div>

        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # RESULT BAR GRAPH
    # --------------------------------------------------------

    with chart_col:

        fig, ax = plt.subplots(figsize=(7, 3.2))

        labels = ["Flu", "No Flu"]
        values = [
            flu_probability * 100,
            no_flu_probability * 100
        ]

        bars = ax.barh(
            labels,
            values,
            height=0.48
        )

        ax.set_xlim(0, 100)

        ax.set_xlabel(
            "Probability (%)",
            color="#334e68",
            fontsize=11
        )

        ax.set_title(
            "Posterior Probability Distribution",
            color="#102a43",
            fontsize=14,
            fontweight="bold"
        )

        ax.tick_params(
            colors="#334e68"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="x",
            alpha=0.2
        )

        for bar, value in zip(bars, values):

            ax.text(
                value + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#102a43"
            )

        fig.tight_layout()

        st.pyplot(fig)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>📈 Probability Update</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Prior</div>
        <div class="metric-value">30.00%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Posterior</div>
        <div class="metric-value">
        {flu_probability * 100:.2f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Evidence Update</div>
        <div class="metric-value">
        {probability_change:+.2f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Uncertainty</div>
        <div class="metric-value" style="font-size:20px;">
        {uncertainty}
        </div>
        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # PRIOR VS POSTERIOR GRAPH
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>🔄 Prior vs Posterior</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))

    labels = ["Prior Probability", "Posterior Probability"]

    values = [
        prior_probability * 100,
        flu_probability * 100
    ]

    bars = ax.bar(
        labels,
        values,
        width=0.48
    )

    ax.set_ylim(0, 100)

    ax.set_ylabel(
        "Probability (%)",
        color="#334e68"
    )

    ax.set_title(
        "How Evidence Changed the Probability of Flu",
        color="#102a43",
        fontsize=14,
        fontweight="bold"
    )

    ax.tick_params(colors="#334e68")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.2
    )

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.2f}%",
            ha="center",
            fontweight="bold",
            color="#102a43"
        )

    fig.tight_layout()

    st.pyplot(fig)


    # --------------------------------------------------------
    # DECISION SUPPORT
    # --------------------------------------------------------

    if flu_probability >= 0.75:

        decision = "High probability of Flu"

        recommendation = (
            "The observed evidence produces a high posterior "
            "probability of Flu. Further clinical evaluation "
            "is recommended."
        )

    elif flu_probability >= 0.50:

        decision = "Moderate probability of Flu"

        recommendation = (
            "The evidence favors Flu, but the probability is "
            "not conclusive. Additional clinical information "
            "would improve the decision."
        )

    else:

        decision = "Lower probability of Flu"

        recommendation = (
            "The current evidence favors No Flu. Additional "
            "symptoms, tests, or medical information may change "
            "the posterior probability."
        )


    st.markdown(
        "<div class='section-title'>🩺 Decision Support</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="decision-card">

    <h3>📌 {decision}</h3>

    <p>{recommendation}</p>

    <p>
    <b>Uncertainty:</b> {uncertainty}
    </p>

    <p>
    <b>Interpretation:</b> {interpretation}
    </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>🔎 Evidence Provided</div>",
        unsafe_allow_html=True
    )

    if evidence:

        evidence_df = pd.DataFrame(
            [
                {
                    "Symptom": symptom,
                    "Observed Value": value
                }
                for symptom, value in evidence.items()
            ]
        )

        st.dataframe(
            evidence_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No symptoms were provided. "
            "The system is using the prior probability."
        )


    # --------------------------------------------------------
    # BAYESIAN REASONING
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>🧮 Bayesian Reasoning</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>Why is the result probabilistic?</h3>

    <p>
    The system does not use a fixed rule such as
    <b>Fever + Cough = Flu</b>.
    </p>

    <div class="info-box">

    <b>Bayesian reasoning:</b><br><br>

    Posterior Probability
    ∝
    Likelihood × Prior Probability

    <br><br>

    <b>
    P(Flu | Evidence)
    =
    P(Evidence | Flu) × P(Flu)
    /
    P(Evidence)
    </b>

    </div>

    <p>
    The evidence updates the initial probability of Flu,
    producing a new posterior probability.
    </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # EVIDENCE CONTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>🧠 Evidence Contribution Analysis</div>",
        unsafe_allow_html=True
    )

    contribution_rows = []

    for symptom, value in evidence.items():

        reduced_evidence = evidence.copy()

        reduced_evidence.pop(symptom)

        reduced_result = inference.query(
            variables=["Flu"],
            evidence=(
                reduced_evidence
                if reduced_evidence
                else None
            )
        )

        reduced_index = (
            reduced_result.state_names["Flu"].index("Yes")
        )

        reduced_probability = float(
            reduced_result.values[reduced_index]
        )

        contribution = (
            flu_probability -
            reduced_probability
        ) * 100

        contribution_rows.append({
            "Evidence": f"{symptom} = {value}",
            "With Evidence": f"{flu_probability * 100:.2f}%",
            "Without Evidence": f"{reduced_probability * 100:.2f}%",
            "Impact": f"{contribution:+.2f} percentage points"
        })

    if contribution_rows:

        contribution_df = pd.DataFrame(
            contribution_rows
        )

        st.dataframe(
            contribution_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Impact is estimated by removing one observed symptom "
            "and recalculating the posterior probability."
        )


    # ========================================================
    # BAYESIAN NETWORK DIAGRAM
    # ========================================================

    st.markdown(
        "<div class='section-title'>🕸️ Bayesian Belief Network</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <p>
    The disease variable is the parent node and the observed
    symptoms are child variables. Evidence entered by the user
    is propagated through this network during probabilistic inference.
    </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # CUSTOM BBN DIAGRAM
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    ax.axis("off")

    # ---- Flu node ----

    flu_box = FancyBboxPatch(
        (4.3, 4.0),
        3.4,
        1.3,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        linewidth=2,
        edgecolor="#1d6fb8",
        facecolor="#e8f3ff"
    )

    ax.add_patch(flu_box)

    ax.text(
        6,
        4.75,
        "FLU",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#102a43"
    )

    ax.text(
        6,
        4.35,
        "Prior probability: 30%",
        ha="center",
        va="center",
        fontsize=11,
        color="#526579"
    )


    # ---- Symptom nodes ----

    nodes = [
        (0.8, 1.8, "FEVER"),
        (4.3, 1.8, "COUGH"),
        (7.8, 1.8, "FATIGUE")
    ]

    for x, y, label in nodes:

        box = FancyBboxPatch(
            (x, y),
            3.0,
            1.25,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.8,
            edgecolor="#8aa8c2",
            facecolor="#f5f9fd"
        )

        ax.add_patch(box)

        ax.text(
            x + 1.5,
            y + 0.78,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#102a43"
        )

        ax.text(
            x + 1.5,
            y + 0.40,
            "Observed evidence",
            ha="center",
            va="center",
            fontsize=10,
            color="#526579"
        )


    # ---- Arrows ----

    arrows = [
        (6.0, 4.0, 2.3, 3.05),
        (5.4, 4.0, 5.8, 3.05),
        (6.6, 4.0, 9.3, 3.05)
    ]

    for x1, y1, x2, y2 in arrows:

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2,
            color="#1d6fb8"
        )

        ax.add_patch(arrow)


    # ---- Inference box ----

    inference_box = FancyBboxPatch(
        (3.3, 0.05),
        5.4,
        0.95,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.5,
        edgecolor="#b7c8d8",
        facecolor="#ffffff"
    )

    ax.add_patch(inference_box)

    ax.text(
        6,
        0.55,
        "Variable Elimination → Posterior Probability",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#102a43"
    )

    fig.tight_layout()

    st.pyplot(fig)


    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    st.markdown(
        "<div class='section-title'>📚 Bayesian Model Parameters</div>",
        unsafe_allow_html=True
    )

    with st.expander("📌 Prior Probability of Flu"):

        prior_df = pd.DataFrame({
            "Flu": ["No", "Yes"],
            "Probability": ["70%", "30%"]
        })

        st.dataframe(
            prior_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("🌡️ Conditional Probability Table — Fever"):

        fever_df = pd.DataFrame({
            "Flu": ["No", "Yes"],
            "P(Fever=No)": ["80%", "20%"],
            "P(Fever=Yes)": ["20%", "80%"]
        })

        st.dataframe(
            fever_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("😷 Conditional Probability Table — Cough"):

        cough_df = pd.DataFrame({
            "Flu": ["No", "Yes"],
            "P(Cough=No)": ["75%", "25%"],
            "P(Cough=Yes)": ["25%", "75%"]
        })

        st.dataframe(
            cough_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("😴 Conditional Probability Table — Fatigue"):

        fatigue_df = pd.DataFrame({
            "Flu": ["No", "Yes"],
            "P(Fatigue=No)": ["70%", "30%"],
            "P(Fatigue=Yes)": ["30%", "70%"]
        })

        st.dataframe(
            fatigue_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # INFERENCE CASES
    # ========================================================

    st.markdown(
        "<div class='section-title'>🧪 Probabilistic Inference Cases</div>",
        unsafe_allow_html=True
    )

    cases = [
        {
            "Case": "Case 1",
            "Evidence": {
                "Fever": "Yes",
                "Cough": "Yes",
                "Fatigue": "No"
            }
        },
        {
            "Case": "Case 2",
            "Evidence": {
                "Fever": "Yes",
                "Cough": "No",
                "Fatigue": "No"
            }
        },
        {
            "Case": "Case 3",
            "Evidence": {
                "Fever": "No",
                "Cough": "No",
                "Fatigue": "No"
            }
        },
        {
            "Case": "Case 4",
            "Evidence": {
                "Fever": "Yes",
                "Cough": "Yes",
                "Fatigue": "Yes"
            }
        }
    ]

    case_rows = []

    for case in cases:

        case_result = inference.query(
            variables=["Flu"],
            evidence=case["Evidence"]
        )

        case_index = (
            case_result.state_names["Flu"].index("Yes")
        )

        probability = float(
            case_result.values[case_index]
        )

        case_rows.append({
            "Case": case["Case"],
            "Evidence": ", ".join(
                f"{key}={value}"
                for key, value in case["Evidence"].items()
            ),
            "Posterior Flu Probability":
                f"{probability * 100:.2f}%"
        })

    cases_df = pd.DataFrame(case_rows)

    st.dataframe(
        cases_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("""
    <div class="footer">

    ⚠️ Educational prototype only.
    This system demonstrates Bayesian probabilistic reasoning
    and is not a medical diagnosis tool.

    </div>
    """, unsafe_allow_html=True)