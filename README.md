# EXACT — Neuro-Symbolic Reasoning Engine

**EXACT** (EXplainable Automated Compositional Thinking) is a deterministic neuro-symbolic reasoning engine that solves educational physics problems and propositional logic queries using structured symbolic pipelines — without relying on LLMs for answer generation.

## Philosophy

EXACT follows a strict **deterministic-first** design:

1. **LLM is never the answer generator** — it can only translate natural language → structured IR (optional, OFF by default)
2. **All reasoning is symbolic** — SymPy for algebra, forward chaining for logic, DAG for multi-step derivation
3. **Every answer is traceable** — full derivation trace from input to output

```
Natural Language Question
        ↓
   SemanticCompiler (NL → WorldModel)
        ↓
   VariableCanonicalizer (alias → formula variable)
        ↓
   DerivationGraph / PhysicsSolver / LogicSolver
        ↓
   Verified Answer + Trace
```

## Architecture

### Core Components

| Component | File | Role |
|:---|:---|:---|
| SemanticCompiler | `src/reasoning/semantic_compiler.py` | Text → WorldModel (3-layer: structural + narrative + LLM) |
| NarrativeExtractor | `src/reasoning/narrative_extractor.py` | Token-window extraction for natural language quantities |
| WorldModel | `src/reasoning/world_model.py` | Unified IR for physics + logic problems |
| VariableCanonicalizer | `src/reasoning/variable_canonicalizer.py` | Maps `U→V`, `C→capacitance`, `I→current` |
| DerivationGraph | `src/reasoning/derivation_graph.py` | Multi-step DAG-based physics planner |
| PhysicsSolver | `src/reasoning/physics_solver.py` | SymPy-based formula solver |
| FormulaBank | `src/reasoning/formula_bank.py` | Registry of physics formulas |
| QuestionTranslator | `src/reasoning/question_translator.py` | 3-layer NL→FOL (rule → heuristic → LLM) |
| Ontology | `src/reasoning/ontology.py` | 120+ predicate aliases, MCQ triggers |
| FOLParser | `src/reasoning/parser.py` | Unicode + Python-style FOL parser |
| PremiseGraph | `src/reasoning/premise_graph.py` | Forward chaining with conjunction rules |
| MCQSolver | `src/reasoning/mcq_solver.py` | MCQ scoring via proof derivability |
| LogicSolver | `src/reasoning/logic_solver.py` | Entailment evaluation |
| Orchestrator | `src/reasoning/solver.py` | Routes physics/logic through full pipeline |

### Pipeline Diagrams

**Physics Pipeline:**
```
Raw Text → StructuralExtractor → NarrativeExtractor → OntologyNormalizer
                                      ↓
                                  WorldModel
                                      ↓
                           VariableCanonicalizer
                                      ↓
                              DerivationGraph (DAG)
                                      ↓
                              PhysicsSolver (SymPy)
                                      ↓
                              Answer + Trace
```

**Logic Pipeline:**
```
JSON Item → QuestionTranslator → Ontology → MCQ/Entailment Goal
                                      ↓
                                  WorldModel
                                      ↓
                              FOLParser → PremiseGraph
                                      ↓
                              Forward Chain → Derivable Facts
                                      ↓
                              MCQSolver / LogicSolver
                                      ↓
                              Answer + Trace
```

## Installation

```bash
# Create environment
conda create -n exact python=3.11 -y
conda activate exact

# Install dependencies
pip install -r requirements.txt

# Optional: install Streamlit for UI
pip install streamlit
```

### Requirements

- Python 3.11+
- PyTorch ≥ 2.0 (for optional LLM compiler)
- SymPy ≥ 1.12
- Streamlit (optional, for UI)

## Running

### Interactive UI
```bash
streamlit run app.py
```

### Benchmarks
```bash
# Physics benchmark (50 problems)
python tests/benchmark_physics_dataset.py

# Logic benchmark (50 items, 93 questions)
python tests/test_logic_dataset_benchmark.py
```

### Tests
```bash
pytest tests/ -v
```

### Programmatic Usage
```python
from src.reasoning.solver import EndToEndOrchestrator

orch = EndToEndOrchestrator()

# Physics
result = orch.run_physics("R = 5 Ω, I = 2 A. Find V.")
print(result.answer)  # "10 V"

# Logic
item = {
    "premises-FOL": ["∀x (WT(x) → O(x))", "WT(Alice)"],
    "questions": ["Does Alice qualify as optimized?"],
    "answers": ["Yes"]
}
result = orch.run_logic(item)
print(result.answer)  # "Yes" or "A"
```

## Project Structure

```
EXACT/
├── app.py                          # Streamlit interactive UI
├── run.py                          # Batch inference entry point
├── requirements.txt
├── config/
│   ├── base.yaml                   # LLM config (model path, quantization)
│   ├── physics.yaml                # Physics-specific settings
│   └── logic.yaml                  # Logic-specific settings
├── data/
│   ├── Physics_Problems_Text_Only.csv   # 50 physics problems
│   └── Logic_Based_Educational_Queries.json  # 50 logic items
├── src/
│   └── reasoning/
│       ├── semantic_compiler.py     # Central grounding layer
│       ├── narrative_extractor.py   # Token-window quantity extraction
│       ├── world_model.py           # Unified intermediate representation
│       ├── variable_canonicalizer.py # Variable alias bridging
│       ├── derivation_graph.py      # Multi-step DAG planner
│       ├── physics_solver.py        # SymPy formula solver
│       ├── formula_bank.py          # Formula registry
│       ├── question_translator.py   # NL → FOL target converter
│       ├── ontology.py              # Predicate aliases + MCQ triggers
│       ├── parser.py                # FOL parser
│       ├── premise_graph.py         # Forward chaining engine
│       ├── mcq_solver.py            # MCQ scoring
│       ├── logic_solver.py          # Entailment evaluation
│       ├── solver.py                # EndToEndOrchestrator
│       └── llm_compiler.py          # Optional LLM bridge (OFF by default)
├── tests/
│   ├── benchmark_physics_dataset.py
│   ├── test_logic_dataset_benchmark.py
│   ├── test_semantic_compiler.py
│   ├── test_question_translator.py
│   ├── test_variable_canonicalizer.py
│   └── test_variable_extractor_dataset.py
├── outputs/                         # Benchmark diagnostics (JSON)
└── docs/
    └── system_audit.md
```

## Benchmarks

### Physics Dataset
- 50 educational physics problems (capacitors, Coulomb's law, Ohm's law, power)
- Metrics: extraction success, formula match, tolerance accuracy (1%), exact match
- Current performance: **6% tolerance accuracy**, **18% extraction failure**

### Logic Dataset
- 50 items, 93 total questions (45 MCQ + 48 Yes/No)
- Metrics: premise parse rate, FOL target coverage, MCQ canonicalization, accuracy
- Current performance: **6.5% overall accuracy**, **98% premise parse**, **64.6% FOL target coverage**

### Running Diagnostics
```bash
python tests/benchmark_physics_dataset.py  # → outputs/physics_diagnostics.json
python tests/test_logic_dataset_benchmark.py  # → outputs/logic_diagnostics.json
```

## LLM Compiler (Optional)

The LLM compiler is a **constrained translation-only** bridge:

- **OFF by default** (`use_llm_compiler: false`)
- LLM may ONLY translate NL → structured IR (JSON)
- LLM may NEVER directly answer questions
- Output must conform to strict JSON schemas (`llm_contracts.py`)
- Deterministic solvers remain authoritative

To enable:
```yaml
# config/base.yaml
llm:
  model_path: /path/to/model
  use_4bit: true
```

```python
orch = EndToEndOrchestrator(llm=your_llm, use_llm_compiler=True)
```

## Current Limitations (Honest Assessment)

1. **Multi-body vector composition**: 3-body Coulomb with arbitrary angles only partially supported (scalar magnitude, not full vector decomposition)
2. **Conditional MCQ options**: "If A then B" options are parsed to FOL implications but proof scoring is still heuristic
3. **Logic Yes/No accuracy**: FOL target extraction works for 64.6% of questions, but the proof engine needs deeper contrapositive/resolution support
4. **Narrative extraction**: Handles 80%+ of common patterns; some complex phrasing still fails
5. **Formula bank**: Limited to ~10 core physics formulas; needs expansion for broader coverage

## Future Work

- **Full vector reasoning**: 2D/3D force decomposition with angle components
- **Theorem proving**: Resolution-based prover for deeper logic inference
- **Graph neural retrieval**: Learned formula/premise retrieval from large banks
- **Probabilistic logic**: Soft inference for uncertain premises
- **Knowledge graph**: Structured physics ontology for concept bridging