# tests/test_question_translator.py
"""Tests for QuestionTranslator using real dataset question patterns."""

import pytest
from src.reasoning.question_translator import QuestionTranslator


@pytest.fixture
def translator():
    return QuestionTranslator()


# ── Yes/No questions (entailment) ────────────────────────────────────────────

def test_scholarship_qualification(translator):
    q = "Does Sophia qualify for the university scholarship, according to the premises?"
    r = translator.translate(q)
    assert r.task_type == "entailment"
    assert r.fol_target is not None
    assert "qualifies_for_scholarship" in r.fol_target
    assert "Sophia" in r.fol_target
    assert r.confidence >= 0.7


def test_scholarship_paraphrase_liam(translator):
    q = "Is Liam eligible for the scholarship, according to the premises?"
    r = translator.translate(q)
    assert r.task_type == "entailment"
    assert r.fol_target is not None
    assert "qualifies_for_scholarship" in r.fol_target


def test_scholarship_financial_aid(translator):
    q = "Can Noah receive financial aid based on the given premises?"
    r = translator.translate(q)
    assert r.task_type in ("entailment", "unknown")
    # Should match via heuristic at minimum


def test_is_project_optimized(translator):
    q = "Is the Python project optimized according to the premises?"
    r = translator.translate(q)
    assert r.task_type in ("entailment", "unknown")


# ── MCQ questions ────────────────────────────────────────────────────────────

def test_mcq_which_conclusion(translator):
    q = ("Which conclusion follows with the fewest premises?\n"
         "A. Option one\nB. Option two\nC. Option three\nD. Option four")
    r = translator.translate(q)
    assert r.task_type == "mcq"
    assert r.fol_target is None
    assert r.confidence >= 0.8


def test_mcq_strongest_conclusion(translator):
    q = ("Based on the above premises, which is the strongest conclusion?\n"
         "A. Sophia qualifies for the university scholarship\n"
         "B. Sophia needs a faculty recommendation")
    r = translator.translate(q)
    assert r.task_type == "mcq"


def test_which_of_following(translator):
    q = "Which of the following best describes the outcome?"
    r = translator.translate(q)
    assert r.task_type == "mcq"


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_unknown_question(translator):
    q = "What is the capital of France?"
    r = translator.translate(q)
    # Should not crash, should be unknown or low confidence
    assert r.task_type in ("entailment", "mcq", "unknown")
    assert 0.0 <= r.confidence <= 1.0


def test_result_always_has_explanation(translator):
    for q in [
        "Does Emma qualify for a scholarship?",
        "Which of the following conclusions is valid?\nA. X\nB. Y",
        "Completely unrelated sentence.",
    ]:
        r = translator.translate(q)
        assert r.explanation, f"No explanation for: {q}"
        assert r.source in ("rule", "heuristic", "llm", "failed")
