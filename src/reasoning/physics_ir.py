# src/reasoning/physics_ir.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class PhysicalEntity:
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

@dataclass
class GeometryRelation:
    relation_type: str  # e.g., "perpendicular_bisector", "right_triangle", "distance"
    entities: List[str] = field(default_factory=list)
    value: Optional[float] = None

@dataclass
class Constraint:
    constraint_type: str # e.g., "same_direction", "opposite"
    entities: List[str] = field(default_factory=list)

@dataclass
class SolveTarget:
    name: str
    unit: Optional[str] = None

@dataclass
class PhysicsProblem:
    entities: Dict[str, PhysicalEntity] = field(default_factory=dict)
    relations: List[GeometryRelation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    targets: List[SolveTarget] = field(default_factory=list)
    
    def add_entity(self, name: str, value: float = None, unit: str = None) -> None:
        if name not in self.entities:
            self.entities[name] = PhysicalEntity(name=name, value=value, unit=unit)
        else:
            if value is not None:
                self.entities[name].value = value
            if unit is not None:
                self.entities[name].unit = unit
