# src/reasoning/premise_graph.py

from typing import Dict, List, Optional, Set, Tuple
from collections import deque
from src.reasoning.fol import ASTNode, ForAll, Implication, Predicate, Negation, Conjunction, Variable, Constant

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

            # Rules: ForAll(x, Implication(A, B))
            rules = [n for n in current_nodes
                     if isinstance(n.ast, ForAll)
                     and isinstance(n.ast.statement, Implication)]

            # Facts: atomic Predicate or Negation(Predicate) with Constant args
            facts = [n for n in current_nodes
                     if isinstance(n.ast, (Predicate, Negation))]

            for rule_node in rules:
                rule = rule_node.ast.statement
                condition = rule.premise
                conclusion = rule.conclusion

                # Expand Conjunction conditions: both conjuncts must be in facts
                if isinstance(condition, Conjunction):
                    cond_list = self._flatten_conjunction(condition)
                else:
                    cond_list = [condition]

                for fact_node in facts:
                    fact_ast = fact_node.ast

                    # Only handle single-condition implications for now
                    # (Conjunction handling requires multiple fact matching)
                    if len(cond_list) > 1:
                        # All conjuncts must unify with some existing fact
                        subst = {}
                        all_match = True
                        for cond in cond_list:
                            matched = False
                            for fn in current_nodes:
                                if not isinstance(fn.ast, (Predicate, Negation)):
                                    continue
                                s = self._try_unify(fn.ast, cond)
                                if s is not None:
                                    subst.update(s)
                                    matched = True
                                    break
                            if not matched:
                                all_match = False
                                break
                        if all_match:
                            new_ast = self.apply_subst(conclusion, subst)
                            new_text = str(new_ast)
                            if not any(n.text == new_text for n in self.premises.values()):
                                new_id = f"D{self.pid_counter}"
                                self.pid_counter += 1
                                new_node = self.add_premise(new_id, new_ast, is_given=False)
                                new_node.derived_from = [rule_node.premise_id, fact_node.premise_id]
                                derived_ids.append(new_id)
                                self._used_premises.add(rule_node.premise_id)
                                changed = True
                        continue

                    # Single condition
                    cond_ast = cond_list[0]
                    subst = self._try_unify(fact_ast, cond_ast)
                    if subst is not None:
                        new_ast = self.apply_subst(conclusion, subst)
                        new_text = str(new_ast)
                        if not any(n.text == new_text for n in self.premises.values()):
                            new_id = f"D{self.pid_counter}"
                            self.pid_counter += 1
                            new_node = self.add_premise(new_id, new_ast, is_given=False)
                            new_node.derived_from = [rule_node.premise_id, fact_node.premise_id]
                            derived_ids.append(new_id)
                            self._used_premises.add(rule_node.premise_id)
                            self._used_premises.add(fact_node.premise_id)
                            changed = True

        return derived_ids

    def _flatten_conjunction(self, conj) -> list:
        """Flatten (A ∧ B ∧ C) into [A, B, C]."""
        from src.reasoning.fol import Conjunction
        if isinstance(conj, Conjunction):
            return self._flatten_conjunction(conj.left) + self._flatten_conjunction(conj.right)
        return [conj]

    def _try_unify(self, fact_ast, cond_ast) -> Optional[Dict]:
        """Try to unify a fact with a condition; handles Predicate and Negation."""
        if isinstance(fact_ast, Negation) and isinstance(cond_ast, Negation):
            if isinstance(fact_ast.statement, Predicate) and isinstance(cond_ast.statement, Predicate):
                return self.unify(fact_ast.statement, cond_ast.statement)
        elif isinstance(fact_ast, Predicate) and isinstance(cond_ast, Predicate):
            return self.unify(fact_ast, cond_ast)
        return None


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
