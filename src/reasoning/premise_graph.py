# src/reasoning/premise_graph.py

from typing import Dict, List, Optional, Set, Tuple
from collections import deque
from src.reasoning.fol import ASTNode, ForAll, Implication, Predicate, Negation, Variable, Constant

class PremiseNode:
    def __init__(self, premise_id: str, ast: ASTNode, is_given: bool = True):
        self.premise_id = premise_id
        self.ast = ast
        self.text = str(ast)
        self.is_given = is_given
        self.derived_from: List[str] = []

class PremiseGraph:
    def __init__(self):
        self.premises: Dict[str, PremiseNode] = {}
        self._used_premises: Set[str] = set()
        self.pid_counter = 100

    def add_premise(self, premise_id: str, ast: ASTNode, is_given: bool = True) -> PremiseNode:
        node = PremiseNode(premise_id, ast, is_given)
        self.premises[premise_id] = node
        return node

    def unify(self, pred1: Predicate, pred2: Predicate) -> Optional[Dict[str, Constant]]:
        """Attempt to unify two predicates. Return substitution dict or None."""
        if pred1.name != pred2.name or len(pred1.args) != len(pred2.args):
            return None
        subst = {}
        for a1, a2 in zip(pred1.args, pred2.args):
            if isinstance(a1, Variable) and isinstance(a2, Constant):
                subst[a1.name] = a2
            elif isinstance(a2, Variable) and isinstance(a1, Constant):
                subst[a2.name] = a1
            elif isinstance(a1, Constant) and isinstance(a2, Constant) and a1.name != a2.name:
                return None
        return subst

    def apply_subst(self, ast: ASTNode, subst: Dict[str, Constant]) -> ASTNode:
        """Apply substitution to AST."""
        if isinstance(ast, Predicate):
            new_args = [subst.get(a.name, a) if isinstance(a, Variable) else a for a in ast.args]
            return Predicate(ast.name, new_args)
        elif isinstance(ast, Negation):
            return Negation(self.apply_subst(ast.statement, subst))
        return ast

    def forward_chain(self) -> List[str]:
        derived_ids = []
        changed = True

        while changed:
            changed = False
            current_nodes = list(self.premises.values())
            
            # Find rules: ForAll(x, Implication(A, B))
            rules = [n for n in current_nodes if isinstance(n.ast, ForAll) and isinstance(n.ast.statement, Implication)]
            
            # Find facts: Predicate or Negation(Predicate) where args are Constants
            facts = [n for n in current_nodes if isinstance(n.ast, (Predicate, Negation))]

            for rule_node in rules:
                rule = rule_node.ast.statement
                condition = rule.premise
                conclusion = rule.conclusion

                for fact_node in facts:
                    # Try to unify fact with rule condition
                    fact_ast = fact_node.ast
                    cond_ast = condition
                    
                    # Handle negations matching
                    if isinstance(fact_ast, Negation) and isinstance(cond_ast, Negation):
                        subst = self.unify(fact_ast.statement, cond_ast.statement)
                    elif not isinstance(fact_ast, Negation) and not isinstance(cond_ast, Negation):
                        subst = self.unify(fact_ast, cond_ast)
                    else:
                        subst = None

                    if subst is not None:
                        new_ast = self.apply_subst(conclusion, subst)
                        new_text = str(new_ast)
                        
                        # Check if we already have this
                        exists = any(n.text == new_text for n in self.premises.values())
                        if not exists:
                            new_id = f"D{self.pid_counter}"
                            self.pid_counter += 1
                            new_node = self.add_premise(new_id, new_ast, is_given=False)
                            new_node.derived_from = [rule_node.premise_id, fact_node.premise_id]
                            derived_ids.append(new_id)
                            self._used_premises.add(rule_node.premise_id)
                            self._used_premises.add(fact_node.premise_id)
                            changed = True

        return derived_ids

    def detect_contradiction(self) -> Optional[str]:
        texts = {}
        for pid, node in self.premises.items():
            ast = node.ast
            if isinstance(ast, Predicate):
                texts[str(ast)] = pid
            elif isinstance(ast, Negation):
                base_str = str(ast.statement)
                if base_str in texts:
                    return f"Contradiction: {base_str} and ¬{base_str}"
        return None

    def get_used_premises(self) -> List[str]:
        return [self.premises[pid].text for pid in self._used_premises if pid in self.premises]
