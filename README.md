# EXACT — Explainable Automated Chain-of-Thought

A neural-symbolic reasoning pipeline that combines **Large Language Models** with **formal logic solvers** to produce verifiable, explainable answers.

The system translates natural language questions into **First-Order Logic (FOL)**, solves them with symbolic engines (SymPy), generates proof traces, and produces human-readable explanations — all with built-in verification and retry mechanisms.

---

## Architecture Overview

```
Input Question
      │
      ▼
┌───────────────┐
│ LLM Classifier│  Identify: Physics | Logic | General
└───────┬───────┘
        │
  ┌─────┼────────────┐
  ▼     ▼            ▼
Physics Logic      General
Solver  Solver     Pipeline
  │     │            │
  ▼     ▼            ▼
┌───────────────┐
│   Verifier    │  Numerical Sanity & Unit Consistency Check
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ LLM Explainer │  Generate natural language explanation
└───────┬───────┘
        │
        ▼
  Unified ReasoningOutput
```

---

## Project Structure

```
EXACT/
├── run.py                          # Main entry point (config-driven)
├── requirements.txt                # Python dependencies
├── config/
│   └── base.yaml                   # Runtime configuration (model path, generation params)
├── data/
│   └── sample.json                 # Input dataset (list of questions)
├── models/
│   └── mistral/
│       └── Mistral-7B-Instruct-v0.2/   # Local model weights
├── scripts/
│   └── test_model.py               # Quick model sanity check
├── src/
│   ├── config.py                   # Hardcoded config (fallback)
│   ├── pipeline.py                 # Pipeline orchestration (single/batch mode)
│   ├── orchestrator.py             # Core 6-step reasoning loop with retry
│   ├── evaluator.py                # Multi-metric evaluation (P1/P2/P3)
│   ├── llm/
│   │   ├── models.py               # LLM loading (HuggingFace, GPU auto-detect)
│   │   ├── generate.py             # Generation wrapper with caching
│   │   ├── prompt.py               # Prompt templates (Planner, FOL, Explainer)
│   │   ├── planner.py              # Step decomposition via LLM
│   │   ├── translator.py           # NL → FOL translation via LLM
│   │   ├── explainer.py            # Proof → explanation via LLM
│   │   └── test_llm.py             # LLM integration test
│   ├── reasoning/
│   │   ├── fol.py                  # FOL data structures (Predicate, Implication, Quantifier)
│   │   ├── parser.py               # FOL string parser
│   │   ├── solver.py               # Symbolic solver (SymPy)
│   │   ├── proof.py                # Proof trace generator
│   │   └── verifier.py             # Answer-proof consistency checker
│   └── utils/
│       ├── cache.py                # MD5-based JSON file cache
│       ├── config_loader.py        # YAML config loader with merge support
│       ├── device.py               # GPU detection utilities
│       ├── io.py                   # JSON/JSONL read/write helpers
│       ├── json_utils.py           # Safe JSON extraction from LLM output
│       ├── logging.py              # File + console logger
│       ├── seed.py                 # Reproducibility seed setter
│       └── timer.py                # Execution timer
├── outputs/                        # Generated predictions & metrics
├── cache/                          # LLM response cache (auto-generated)
└── logs/                           # Runtime logs (auto-generated)
```

---

## Quick Start

### 1. Environment Setup

```bash
conda activate exact
pip install -r requirements.txt
```

**Dependencies:** `torch` (CUDA), `transformers`, `accelerate`, `safetensors`, `sentencepiece`, `sympy`, `pyyaml`, `tqdm`

### 2. Run Full Pipeline

```bash
python run.py --config config/base.yaml
```

This will:
- Load dataset from `data/sample.json`
- Load model from the path specified in `config/base.yaml`
- Process each question through the 6-step reasoning pipeline
- Cache results to avoid re-computation
- Evaluate with P1/P2/P3 metrics
- Save predictions to `outputs/predictions.json`
- Save metrics to `outputs/predictions.json.metrics.json`

### 3. Quick Model Test

```bash
python scripts/test_model.py
```

Loads the model and runs a single prompt — useful for verifying GPU setup and model loading.

---

## Configuration

### `config/base.yaml`

```yaml
data_path: data/sample.json        # Input data
task_type: qa                       # Dataset type: qa | fol | physics
output_path: outputs/predictions.json
cache_dir: cache/

llm:
  model_path: /path/to/model        # Absolute path to local HuggingFace model
  use_4bit: true                     # Quantization flag (not yet implemented)
  max_new_tokens: 256                # Max generation length
  temperature: 0.7                   # Sampling temperature
  top_p: 0.9                         # Nucleus sampling threshold
```

---

## Key Components

### Orchestrator (`src/orchestrator.py`)

The central reasoning loop classifies questions and routes them to appropriate solvers:

| Solver | Used For | Mechanism |
|--------|----------|-----------|
| **Physics Solver** | Numerical physics problems | Hybrid regex/LLM variable extraction + Formula Bank + SymPy computation |
| **Logic Solver** | Deductive logic problems | LLM premise extraction + Premise Graph + Forward Chaining |
| **General Solver** | Fallback & Definitions | LLM step planning + Best-effort FOL translation |

The output from any solver is passed to the **Verifier** for multi-layer validation (numerical sanity, unit consistency, reasoning consistency). If verified (or high confidence), the **LLM Explainer** generates a human-readable explanation.

### Evaluator (`src/evaluator.py`)

Four evaluation metrics, each normalized to `[0, 1]`:

| Metric | What it measures |
|--------|------------------|
| **P1 — Accuracy** | Output match against reference (with numerical tolerance) |
| **P2 — Explanation** | Heuristic quality: length, reasoning keywords, structure |
| **P3 — Reasoning** | Presence of structured artifacts: formal representation, proof steps, premise tracking |
| **P4 — Trace** | Completeness of the step-by-step reasoning trace based on the question type |

### LLM Layer (`src/llm/`)

- **`models.py`** — Loads HuggingFace models with `device_map="auto"` for multi-GPU, `float16` precision
- **`generate.py`** — Wraps generation with MD5-based file caching to avoid redundant API/inference calls
- **`prompt.py`** — Structured JSON prompt templates that enforce output format

### Data Layer (`src/data/`)

Supports 3 dataset types via `task_type` config:

| Type | Schema | Fields |
|------|--------|--------|
| `qa` | `QASample` | `question`, `answer`, `explanation` |
| `fol` | `FOLSample` | + `fol` (ground truth FOL) |
| `physics` | `PhysicsSample` | + `reasoning`, `formula` |

### Input Format

```json
[
    {"question": "What is Ohm's law?"},
    {"question": "Calculate current if V=10V and R=5 ohm.", "answer": "2A"}
]
```

---

## GPU Support

The codebase auto-detects CUDA and configures accordingly:

| Setting | GPU Available | CPU Fallback |
|---------|--------------|--------------|
| Model dtype | `float16` | `float32` |
| Device mapping | `device_map="auto"` (multi-GPU) | Manual `.to("cpu")` |
| Input tensors | Sent to `model.device` | CPU |

**Tested hardware:** RTX 3090 Ti (24GB), TITAN V (12GB), RTX 3060 (12GB)

---

## Output Format

Each prediction is a dict:

```json
{
    "question": "Calculate current if V=10V and R=5 ohm.",
    "steps": ["Identify known variables", "Apply Ohm's law: I = V/R"],
    "fol": "Implies(And(V=10, R=5), I=V/R)",
    "parsed": {"type": "implication", "premise": "...", "conclusion": "..."},
    "answer": "2",
    "proof": ["Given premise: ...", "If premise holds, then: ...", "Result: 2"],
    "explanation": "Using Ohm's law, I = V/R = 10/5 = 2A.",
    "valid": true,
    "attempt": 0
}
```