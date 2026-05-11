# src/reasoning/world_model.py
"""
Unified Intermediate Representation (World Model) for EXACT.

Both physics and logic problems are represented through this schema.
Solvers MUST accept WorldModel; they must NOT parse raw text directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ──────────────────────────────────────────────
# SHARED PRIMITIVES
# ──────────────────────────────────────────────

@dataclass
class Quantity:
    """A physical measurement with optional unit and source tracking."""
    value: float
    unit: Optional[str] = None
    # Source: "explicit" (from text), "inferred" (geometry), "constant" (k, etc.)
    source: str = "explicit"
    raw_text: Optional[str] = None   # original string for audit trail

    def __repr__(self):
        s = f"{self.value:.4g}"
        if self.unit:
            s += f" {self.unit}"
        return s


@dataclass
class Entity:
    """A named object in the problem (charge, resistor, person, ...)."""
    name: str                            # canonical name: q1, Sophia, C1
    entity_type: str = "unknown"         # "charge", "resistor", "person", ...
    attributes: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"charge": Quantity(6e-8, "C"), "sign": "positive"}

    def set_attr(self, key: str, value: Any):
        self.attributes[key] = value

    def get_attr(self, key: str, default=None):
        return self.attributes.get(key, default)


@dataclass
class Relation:
    """A binary or n-ary relation between entities."""
    relation_type: str   # "distance", "implies", "between", "triangle", ...
    entities: List[str]  # entity names involved
    value: Optional[Quantity] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryConstraint:
    """Spatial constraint reconstructed from text."""
    constraint_type: str   # "right_triangle", "equilateral_triangle",
                           # "perpendicular_bisector", "collinear", "angle"
    points: List[str] = field(default_factory=list)
    value: Optional[float] = None   # angle in degrees or None


@dataclass
class Goal:
    """What the problem asks us to compute or prove."""
    goal_type: str    # "compute", "prove", "mcq", "entailment"
    target: str       # variable name (physics) or FOL string (logic)
    unit: Optional[str] = None
    options: List[MCQOption] = field(default_factory=list)   # populated for MCQ


@dataclass
class MCQOption:
    """A single MCQ answer option."""
    option_id: str         # "A", "B", "C", "D"
    text: str              # raw NL text
    fol_str: Optional[str] = None    # canonicalized FOL, if available
    score: float = 0.0


# ──────────────────────────────────────────────
# DERIVED FACT (output of reasoning engines)
# ──────────────────────────────────────────────

@dataclass
class DerivedFact:
    fact_type: str         # "quantity", "predicate", "relation"
    name: str
    value: Any             # Quantity for physics, str for logic predicates
    derived_by: str        # formula_id or rule_id
    confidence: float = 1.0


# ──────────────────────────────────────────────
# WORLD MODEL — unified root object
# ──────────────────────────────────────────────

@dataclass
class WorldModel:
    """
    The canonical intermediate representation for a single reasoning problem.
    After SemanticCompiler builds this, solvers operate ONLY on WorldModel.
    """
    # Source tracking
    raw_text: str = ""
    problem_type: str = "unknown"   # "physics" | "logic" | "unknown"

    # Entities present in the problem
    entities: Dict[str, Entity] = field(default_factory=dict)

    # Relations between entities
    relations: List[Relation] = field(default_factory=list)

    # Geometric constraints (physics)
    geometry: List[GeometryConstraint] = field(default_factory=list)

    # Logic premises (FOL strings or AST refs)
    premises_fol: List[str] = field(default_factory=list)

    # Problem goals
    goals: List[Goal] = field(default_factory=list)

    # Output: facts derived during reasoning
    derived_facts: List[DerivedFact] = field(default_factory=list)

    # Compilation metadata
    compilation_method: str = "unknown"  # "structural", "llm", "hybrid"
    compilation_confidence: float = 0.0

    # ── Convenience helpers ──

    def add_entity(self, name: str, entity_type: str = "unknown") -> Entity:
        if name not in self.entities:
            self.entities[name] = Entity(name=name, entity_type=entity_type)
        return self.entities[name]

    def set_quantity(self, entity_name: str, attr: str,
                     value: float, unit: Optional[str] = None,
                     source: str = "explicit"):
        ent = self.add_entity(entity_name)
        ent.set_attr(attr, Quantity(value, unit, source))

    def add_relation(self, rel_type: str, entities: List[str],
                     value: Optional[Quantity] = None, **meta):
        self.relations.append(Relation(rel_type, entities, value, meta))

    def add_geometry(self, constraint_type: str, points: List[str],
                     value: Optional[float] = None):
        self.geometry.append(GeometryConstraint(constraint_type, points, value))

    def get_quantity(self, entity_name: str, attr: str) -> Optional[Quantity]:
        ent = self.entities.get(entity_name)
        if not ent:
            return None
        return ent.get_attr(attr)

    def flat_quantities(self) -> Dict[str, float]:
        """
        Flatten all entity quantities into a single {name: float} dict.
        For single-entity quantities (like V=10, R=5) this mirrors the old
        VariableExtractor output, so PhysicsSolver can use it without refactor.
        """
        result: Dict[str, float] = {}
        for ent_name, ent in self.entities.items():
            for attr_key, attr_val in ent.attributes.items():
                if isinstance(attr_val, Quantity):
                    # Use entity name if attr_key is the entity type qualifier
                    key = ent_name if attr_key in ("value", "quantity") else f"{ent_name}_{attr_key}"
                    # Special case: single-value entities map directly
                    if len(ent.attributes) == 1:
                        key = ent_name
                    result[key] = attr_val.value
        return result

    def primary_goal_target(self) -> Optional[str]:
        for g in self.goals:
            if g.goal_type in ("compute", "prove", "entailment"):
                return g.target
        return None

    def has_geometry(self, gtype: str) -> bool:
        return any(g.constraint_type == gtype for g in self.geometry)
