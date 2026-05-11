# src/reasoning/template_retriever.py

from typing import List, Dict, Any
from src.reasoning.physics_ir import PhysicsProblem
from src.reasoning.formula_bank import FORMULA_BANK

class TemplateRetriever:
    """
    Symbolic Retrieval bypassing vector DBs.
    Uses structural fingerprints and keyword signatures to retrieve formula templates.
    """
    
    def retrieve_physics_templates(self, problem: PhysicsProblem) -> List[str]:
        """
        Returns a ranked list of formula IDs based on IR structural signatures.
        """
        signatures = set()
        
        # 1. Geometry signatures
        for rel in problem.relations:
            signatures.add(rel.relation_type)
            
        # 2. Constraint signatures
        for con in problem.constraints:
            signatures.add(con.constraint_type)
            
        # 3. Entity signatures
        for ent in problem.entities.values():
            if ent.unit == "C" or ent.name in ["q", "q1", "q2", "charge"]:
                signatures.add("electrostatics")
            if ent.unit == "F" or ent.name in ["C", "capacitance"]:
                signatures.add("capacitor")
                
        # Matching
        ranked = []
        for fid, formula in FORMULA_BANK.items():
            score = 0
            # Match aliases and keywords directly against signatures
            formula_sigs = set(formula.get("keywords", []) + formula.get("aliases", []))
            
            if "right_triangle" in signatures and "right triangle" in formula_sigs:
                score += 5
            if "equilateral_triangle" in signatures and "equilateral" in formula_sigs:
                score += 5
            if "electrostatics" in signatures and "coulomb" in formula_sigs:
                score += 2
            if "capacitor" in signatures and "capacitor" in formula_sigs:
                score += 2
                
            if score > 0:
                ranked.append((fid, score))
                
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [fid for fid, score in ranked]
