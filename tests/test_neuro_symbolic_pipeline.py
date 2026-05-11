# tests/test_neuro_symbolic_pipeline.py

import pytest
from src.reasoning.physics_ir import PhysicsProblem
from src.reasoning.physics_semantic_parser import PhysicsSemanticParser
from src.reasoning.template_retriever import TemplateRetriever
from src.reasoning.physics_planner import PhysicsPlanner

def test_physics_semantic_parser():
    parser = PhysicsSemanticParser()
    text = "Three charges q1 = 5 μC, q2 = -5 uC, and q3 = 4 μC form a right-angled triangle. Find the force."
    ir = parser.parse(text)
    
    # 1. Check Unicode and prefix normalization
    assert "charge" in ir.entities or "q1" in ir.entities # The extractor catches q1, q2, q3 if aliases are set
    
    # 2. Check structural heuristic
    assert any(rel.relation_type == "right_triangle" for rel in ir.relations)
    
    # 3. Target
    assert len(ir.targets) > 0
    assert ir.targets[0].name == "F"

def test_template_retrieval():
    ir = PhysicsProblem()
    parser = PhysicsSemanticParser()
    ir = parser.parse("equilateral triangle with charges q1 = 1 C and q2 = 1 C")
    
    retriever = TemplateRetriever()
    templates = retriever.retrieve_physics_templates(ir)
    
    # Should boost equilateral electrostatics
    assert "equilateral_triangle_electrostatics" in templates[:3]

def test_physics_planner_dag():
    ir = PhysicsProblem()
    # Let's say we have V and I, we want to find energy over some time (if we had time).
    # Or we have V, R, we want P. Power_ir is P = I^2 * R. But we have V, R.
    # We must use ohm_law (I = V/R) then power_ir.
    
    ir.add_entity("V", 10.0)
    ir.add_entity("R", 2.0)
    
    from src.reasoning.physics_ir import SolveTarget
    ir.targets.append(SolveTarget(name="P"))
    
    planner = PhysicsPlanner()
    plan = planner.plan_derivation(ir)
    
    # Should be a two step plan: solve current from ohm_law, then P from power_vi or power_ir
    # Actually power_vi is P=V*current. So if we have V, we need current.
    assert len(plan) > 0
    
    result = planner.execute_plan(ir, plan)
    assert "P" in result
    assert result["P"] == 50.0 # V=10, R=2 -> I=5. P = 10*5 = 50.
