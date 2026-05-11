# src/reasoning/physics_semantic_parser.py

import re
from typing import Dict, List, Optional
from src.reasoning.physics_ir import PhysicsProblem, PhysicalEntity, GeometryRelation, Constraint, SolveTarget
from src.reasoning.variable_extractor import VariableExtractor

class PhysicsSemanticParser:
    """
    Parses natural language physics narratives into structured PhysicsProblem IR.
    Uses token windows and dependency-aware heuristics, bypassing LLM hallucinations.
    """
    def __init__(self):
        self.extractor = VariableExtractor()
        
    def _extract_geometry(self, text: str, problem: PhysicsProblem):
        text_lower = text.lower()
        
        # Simple token window heuristics for geometry
        if "equilateral triangle" in text_lower:
            problem.relations.append(GeometryRelation("equilateral_triangle"))
        if "right-angled triangle" in text_lower or "right angle" in text_lower or "orthogonal" in text_lower:
            problem.relations.append(GeometryRelation("right_triangle"))
        if "perpendicular bisector" in text_lower:
            problem.relations.append(GeometryRelation("perpendicular_bisector"))
            
        # Parse distances between specific points: "AB = 4 m" or "AC = 12 cm"
        dist_matches = re.finditer(r'([A-Z]{2})\s*=\s*([\d.]+)\s*([cmk]?m)', text)
        for match in dist_matches:
            points = match.group(1)
            val = float(match.group(2))
            unit = match.group(3)
            
            # Normalize to meters
            if unit == 'cm': val *= 1e-2
            elif unit == 'km': val *= 1e3
            elif unit == 'mm': val *= 1e-3
            
            problem.relations.append(GeometryRelation(
                relation_type="distance",
                entities=[points[0], points[1]],
                value=val
            ))

    def _extract_constraints(self, text: str, problem: PhysicsProblem):
        text_lower = text.lower()
        if "same direction" in text_lower:
            problem.constraints.append(Constraint("same_direction"))
        if "opposite direction" in text_lower:
            problem.constraints.append(Constraint("opposite_direction"))

    def parse(self, text: str) -> PhysicsProblem:
        problem = PhysicsProblem()
        
        # 1. Base Variables via robust Extractor
        extracted_vars = self.extractor.extract(text)
        for name, value in extracted_vars.items():
            problem.add_entity(name=name, value=value)
            
        # 2. Extract Narrative Specific Entities (q1, q2, q3) placed at points
        # e.g. "charges q1 = 5 uC, q2 = -5 uC are placed at A and B"
        # Since the extractor catches q1 = 5uC, we just need to associate points if needed.
        # For this level, storing them as entities is sufficient.
        
        # 3. Semantic Geometry Parsing
        self._extract_geometry(text, problem)
        
        # 4. Constraints
        self._extract_constraints(text, problem)
        
        # 5. Targets
        target_name = self.extractor.extract_target(text)
        if target_name and target_name != "Unknown":
            problem.targets.append(SolveTarget(name=target_name))
            
        return problem
