# app.py
"""
EXACT — Interactive Neuro-Symbolic Reasoning Interface (Streamlit)

Launch: streamlit run app.py
"""

import streamlit as st
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.reasoning.solver import EndToEndOrchestrator
from src.reasoning.semantic_compiler import SemanticCompiler
from src.reasoning.variable_canonicalizer import VariableCanonicalizer
from src.reasoning.derivation_graph import DerivationGraph


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EXACT — Neuro-Symbolic Reasoning",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .result-box { background: #1a1a2e; color: #e0e0e0; padding: 1.2rem;
                  border-radius: 8px; border-left: 4px solid #00d4ff;
                  margin: 0.8rem 0; font-family: monospace; }
    .trace-step { background: #16213e; padding: 0.8rem; border-radius: 6px;
                  margin: 0.3rem 0; font-size: 0.85rem; }
    .metric-card { background: #0f3460; padding: 1rem; border-radius: 8px;
                   text-align: center; }
    .confidence-high { color: #00ff88; }
    .confidence-low  { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_orchestrator():
    return EndToEndOrchestrator()


@st.cache_resource
def get_compiler():
    return SemanticCompiler()


@st.cache_resource
def get_canonicalizer():
    return VariableCanonicalizer()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ EXACT Engine")
mode = st.sidebar.radio("Mode", ["Physics", "Logic"], index=0)
debug = st.sidebar.checkbox("Show debug trace", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Examples")

if mode == "Physics":
    examples = {
        "Capacitor energy": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V.",
        "Ohm's law": "Given R = 10 Ω and V = 30 V, find the current.",
        "Coulomb 2-body": "Two charges q1 = 6 × 10^-8 C and q2 = -6 × 10^-8 C are 8 cm apart. Find the force.",
        "Power (V,I)": "A circuit has voltage V = 120 V and current I = 5 A. Calculate the power.",
        "Narrative voltage": "The voltage is 120 V and the resistance is 60 ohms. What is the current?",
    }
else:
    examples = {
        "Scholarship (Yes/No)": json.dumps({
            "premises-FOL": [
                "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
                "ForAll(x, (qualified_for_advanced_courses(x) ∧ completed_research_methodology(x)) → eligible_for_international_program(x))",
                "ForAll(x, (eligible_for_international_program(x) ∧ completed_capstone_project(x)) → awarded_honors_diploma(x))",
                "ForAll(x, (awarded_honors_diploma(x) ∧ completed_community_service(x)) → qualifies_for_scholarship(x))",
                "completed_core_curriculum(Sophia)", "passed_science_assessment(Sophia)",
                "completed_research_methodology(Sophia)", "completed_capstone_project(Sophia)",
                "completed_community_service(Sophia)"
            ],
            "questions": ["Does Sophia qualify for the university scholarship, according to the premises?"],
            "answers": ["Yes"]
        }, indent=2),
        "Software quality MCQ": json.dumps({
            "premises-FOL": [
                "∀x (WT(x) → O(x))", "∀x (WT(x))", "∀x (O(x) → CR(x))"
            ],
            "questions": [
                "Which conclusion follows?\nA. If a project is well-tested, it has clean code\nB. Projects are not optimized\nC. Testing is irrelevant\nD. No project has clean code"
            ],
            "answers": ["A"]
        }, indent=2),
    }

selected_example = st.sidebar.selectbox("Load example", ["(none)"] + list(examples.keys()))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚡ EXACT — Neuro-Symbolic Reasoning Engine")
st.markdown("Deterministic reasoning with symbolic solvers. LLM is OFF by default.")

orchestrator = get_orchestrator()
compiler = get_compiler()
canon = get_canonicalizer()


if mode == "Physics":
    st.header("🔬 Physics Mode")

    default_text = ""
    if selected_example != "(none)" and selected_example in examples:
        default_text = examples[selected_example]

    question = st.text_area("Enter physics problem:", value=default_text, height=100,
                            placeholder="e.g. Two charges q1 = 6e-8 C and q2 = -6e-8 C, 8 cm apart. Find the force.")

    if st.button("Solve", type="primary") and question.strip():
        with st.spinner("Compiling world model..."):
            wm = compiler.compile_physics(question)
            result = orchestrator.run_physics(question)

        col1, col2, col3 = st.columns(3)
        with col1:
            conf_class = "confidence-high" if result.confidence > 0.5 else "confidence-low"
            st.markdown(f"### Answer\n**{result.answer}**")
        with col2:
            st.metric("Confidence", f"{result.confidence:.0%}")
        with col3:
            st.metric("Valid", "✅" if result.valid else "❌")

        st.markdown(f"**Explanation:** {result.explanation}")

        if debug:
            with st.expander("📊 Extracted Variables", expanded=True):
                flat = wm.flat_quantities()
                canonical = canon.normalize_for_formula(flat)
                st.json({"raw": flat, "canonical": canonical})

            with st.expander("🌐 World Model"):
                st.json({
                    "entities": {k: str(v) for k, v in wm.entities.items()},
                    "relations": [str(r) for r in wm.relations],
                    "goals": [str(g) for g in wm.goals],
                    "compilation": wm.compilation_method,
                })

            with st.expander("🔍 Reasoning Trace"):
                for step in result.reasoning_trace:
                    st.markdown(f'<div class="trace-step">{json.dumps(step, default=str)}</div>',
                                unsafe_allow_html=True)


elif mode == "Logic":
    st.header("🧠 Logic Mode")

    default_json = ""
    if selected_example != "(none)" and selected_example in examples:
        default_json = examples[selected_example]

    raw_input = st.text_area("Enter logic problem (JSON):", value=default_json, height=250,
                             placeholder='{"premises-FOL": [...], "questions": [...]}')

    if st.button("Reason", type="primary") and raw_input.strip():
        try:
            item = json.loads(raw_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        with st.spinner("Running inference..."):
            wm = compiler.compile_logic(item)
            result = orchestrator.run_logic(item)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### Answer\n**{result.answer}**")
        with col2:
            st.metric("Confidence", f"{result.confidence:.0%}")
        with col3:
            expected = item.get("answers", [None])[0]
            if expected:
                correct = result.answer and result.answer.upper() == str(expected).upper()
                st.metric("Correct", "✅" if correct else "❌")

        st.markdown(f"**Explanation:** {result.explanation}")

        if debug:
            with st.expander("📜 Premises FOL", expanded=True):
                for i, p in enumerate(item.get("premises-FOL", [])):
                    st.code(p, language="text")

            with st.expander("🎯 Goals"):
                for g in wm.goals:
                    st.json({"type": g.goal_type, "target": g.target,
                             "options": [{"id": o.option_id, "text": o.text, "fol": o.fol_str}
                                         for o in g.options] if g.options else []})

            with st.expander("🔍 Reasoning Trace"):
                for step in result.reasoning_trace:
                    st.markdown(f'<div class="trace-step">{json.dumps(step, default=str)}</div>',
                                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("**EXACT v2.2** | Neuro-Symbolic Engine")
st.sidebar.markdown("LLM Compiler: **OFF**")
st.sidebar.caption("All reasoning is deterministic.")
