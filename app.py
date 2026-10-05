import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Bayesian Medical Expert System",
    page_icon="🧠",
    layout="wide"
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🧠 Medical Diagnosis Expert System")
st.subheader("Bayesian Belief Network based Probabilistic Inference")

st.info(
    "This Expert System uses a Bayesian Belief Network (BBN) "
    "to handle uncertainty and calculate the probability of Flu "
    "from observed symptoms."
)


# -------------------------------------------------
# CREATE BAYESIAN NETWORK
# -------------------------------------------------

model = DiscreteBayesianNetwork([
    ("Flu", "Fever"),
    ("Flu", "Cough"),
    ("Flu", "Fatigue")
])


# -------------------------------------------------
# PRIOR PROBABILITY
# P(Flu)
# -------------------------------------------------

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


# -------------------------------------------------
# CONDITIONAL PROBABILITY
# P(Fever | Flu)
# -------------------------------------------------

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


# -------------------------------------------------
# P(Cough | Flu)
# -------------------------------------------------

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


# -------------------------------------------------
# P(Fatigue | Flu)
# -------------------------------------------------

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


# Add CPDs to model
model.add_cpds(
    cpd_flu,
    cpd_fever,
    cpd_cough,
    cpd_fatigue
)

# Check model
model.check_model()

# Inference engine
inference = VariableElimination(model)


# -------------------------------------------------
# SIDEBAR - PATIENT INPUT
# -------------------------------------------------

st.sidebar.header("👤 Patient Information")

st.sidebar.write("Select the observed symptoms:")

fever = st.sidebar.selectbox(
    "🌡️ Fever",
    ["No", "Yes"]
)

cough = st.sidebar.selectbox(
    "🤧 Cough",
    ["No", "Yes"]
)

fatigue = st.sidebar.selectbox(
    "😴 Fatigue",
    ["No", "Yes"]
)

diagnose = st.sidebar.button(
    "🔍 Perform Probabilistic Diagnosis",
    use_container_width=True
)


# -------------------------------------------------
# MAIN DIAGNOSIS
# -------------------------------------------------

if diagnose:

    evidence = {
        "Fever": fever,
        "Cough": cough,
        "Fatigue": fatigue
    }

    # ACTUAL BAYESIAN INFERENCE
    result = inference.query(
        variables=["Flu"],
        evidence=evidence
    )

    probability_no_flu = float(
        result.values[result.state_names["Flu"].index("No")]
    )

    probability_flu = float(
        result.values[result.state_names["Flu"].index("Yes")]
    )

    flu_percentage = probability_flu * 100
    no_flu_percentage = probability_no_flu * 100


    # -------------------------------------------------
    # RESULT HEADER
    # -------------------------------------------------

    st.header("📊 Probabilistic Diagnosis Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Probability of Flu",
            f"{flu_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Probability of No Flu",
            f"{no_flu_percentage:.2f}%"
        )

    with col3:

        if flu_percentage >= 70:
            status = "High Probability"
        elif flu_percentage >= 40:
            status = "Moderate Probability"
        else:
            status = "Low Probability"

        st.metric(
            "Uncertainty Level",
            status
        )


    # -------------------------------------------------
    # PROBABILITY BAR
    # -------------------------------------------------

    st.subheader("📈 Posterior Probability")

    st.progress(
        int(flu_percentage)
    )

    st.write(
        f"After considering the observed evidence, "
        f"the calculated probability of Flu is "
        f"**{flu_percentage:.2f}%**."
    )


    # -------------------------------------------------
    # RECOMMENDATION
    # -------------------------------------------------

    st.subheader("💡 Expert System Recommendation")

    if flu_percentage >= 70:

        st.error(
            "High probability of Flu. Further medical evaluation "
            "is recommended."
        )

    elif flu_percentage >= 40:

        st.warning(
            "Moderate probability of Flu. Additional information "
            "or medical evaluation may be required."
        )

    else:

        st.success(
            "Low probability of Flu based on the given evidence."
        )


    # -------------------------------------------------
    # EVIDENCE USED
    # -------------------------------------------------

    st.subheader("🔎 Evidence Used for Inference")

    evidence_col1, evidence_col2, evidence_col3 = st.columns(3)

    with evidence_col1:
        st.write("🌡️ Fever")
        st.success(fever)

    with evidence_col2:
        st.write("🤧 Cough")
        st.success(cough)

    with evidence_col3:
        st.write("😴 Fatigue")
        st.success(fatigue)


    # -------------------------------------------------
    # BAYESIAN EXPLANATION
    # -------------------------------------------------

    with st.expander("🧮 How did the system calculate this?"):

        st.write(
            "The system uses Bayesian probabilistic inference."
        )

        st.latex(
            r"P(Flu|Evidence) \propto P(Flu) \times P(Evidence|Flu)"
        )

        st.write(
            "The prior probability of Flu is updated using "
            "the observed symptoms. The inference engine "
            "calculates the posterior probability."
        )

        st.write(
            "Therefore, the answer is not a fixed rule such as "
            "'Fever + Cough = Flu'. The probability changes "
            "according to the evidence."
        )


# -------------------------------------------------
# BBN VISUALIZATION
# -------------------------------------------------

st.header("🔗 Bayesian Belief Network")

st.write(
    "The following network represents the probabilistic "
    "relationships used by the Expert System."
)

graph = nx.DiGraph()

graph.add_edges_from([
    ("Flu", "Fever"),
    ("Flu", "Cough"),
    ("Flu", "Fatigue")
])

fig, ax = plt.subplots(figsize=(10, 4))

position = {
    "Flu": (0, 0),
    "Fever": (-2, -1),
    "Cough": (0, -1),
    "Fatigue": (2, -1)
}

nx.draw(
    graph,
    position,
    with_labels=True,
    node_size=4000,
    node_color="lightblue",
    font_size=12,
    font_weight="bold",
    arrows=True,
    ax=ax
)

st.pyplot(fig)


# -------------------------------------------------
# PROBABILITY TABLES
# -------------------------------------------------

st.header("📋 Probability Information")

with st.expander("Prior Probability of Flu"):

    st.write("P(Flu = No) = 0.70")
    st.write("P(Flu = Yes) = 0.30")


with st.expander("Conditional Probability Tables"):

    st.write("### P(Fever | Flu)")

    st.dataframe(
        {
            "Flu": ["No", "Yes"],
            "Fever = No": [0.80, 0.20],
            "Fever = Yes": [0.20, 0.80]
        },
        use_container_width=True
    )

    st.write("### P(Cough | Flu)")

    st.dataframe(
        {
            "Flu": ["No", "Yes"],
            "Cough = No": [0.75, 0.25],
            "Cough = Yes": [0.25, 0.75]
        },
        use_container_width=True
    )

    st.write("### P(Fatigue | Flu)")

    st.dataframe(
        {
            "Flu": ["No", "Yes"],
            "Fatigue = No": [0.70, 0.30],
            "Fatigue = Yes": [0.30, 0.70]
        },
        use_container_width=True
    )


# -------------------------------------------------
# DEMONSTRATION OF PROBABILISTIC INFERENCE
# -------------------------------------------------

st.header("🧪 Probabilistic Inference Demonstration")

st.write(
    "Different evidence produces different posterior probabilities. "
    "This demonstrates uncertainty handling."
)

cases = [
    {
        "Case": "Case 1",
        "Fever": "Yes",
        "Cough": "Yes",
        "Fatigue": "No"
    },
    {
        "Case": "Case 2",
        "Fever": "Yes",
        "Cough": "No",
        "Fatigue": "No"
    },
    {
        "Case": "Case 3",
        "Fever": "No",
        "Cough": "No",
        "Fatigue": "No"
    },
    {
        "Case": "Case 4",
        "Fever": "Yes",
        "Cough": "Yes",
        "Fatigue": "Yes"
    }
]

case_results = []

for case in cases:

    case_evidence = {
        "Fever": case["Fever"],
        "Cough": case["Cough"],
        "Fatigue": case["Fatigue"]
    }

    case_result = inference.query(
        variables=["Flu"],
        evidence=case_evidence
    )

    flu_probability = float(
        case_result.values[
            case_result.state_names["Flu"].index("Yes")
        ]
    )

    case_results.append({
        "Case": case["Case"],
        "Fever": case["Fever"],
        "Cough": case["Cough"],
        "Fatigue": case["Fatigue"],
        "Probability of Flu": f"{flu_probability * 100:.2f}%"
    })


st.dataframe(
    case_results,
    use_container_width=True
)


# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------

st.warning(
    "⚠️ This is an academic prototype demonstrating "
    "Bayesian probabilistic reasoning. The probabilities "
    "are illustrative and should not be used for real medical diagnosis."
)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.caption(
    "Experiment 10 | Expert System using Bayesian Belief Network | "
    "Probabilistic Inference"
)