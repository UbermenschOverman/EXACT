# src/reasoning/physics_scene_builder.py
"""
Reconstructs a physics scene from a WorldModel.
Handles triangle geometry, coordinate assignment, distance derivation,
and force direction inference.
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from src.reasoning.world_model import WorldModel, GeometryConstraint, Quantity


class PhysicsSceneBuilder:
    """
    Given a (partially-filled) WorldModel, enriches it with:
    - Detected triangle geometry
    - Derived distances from geometric constraints
    - Angle derivation from triangle types
    """

    def enrich(self, wm: WorldModel) -> WorldModel:
        """Main entry point. Returns the same WorldModel, enriched."""
        text = wm.raw_text.lower()

        self._detect_triangle_type(wm, text)
        self._extract_point_distances(wm, wm.raw_text)
        self._derive_triangle_distances(wm)
        self._infer_angles_from_geometry(wm)
        return wm

    # ── Triangle type detection ──────────────────────────────────────

    def _detect_triangle_type(self, wm: WorldModel, text: str):
        if "equilateral triangle" in text:
            wm.add_geometry("equilateral_triangle", [])
        elif "right-angled triangle" in text or "right angle" in text or "orthogonal" in text:
            wm.add_geometry("right_triangle", [])
        elif "isosceles" in text:
            wm.add_geometry("isosceles_triangle", [])

        if "perpendicular bisector" in text:
            wm.add_geometry("perpendicular_bisector", [])

        # Detect "AC² + BC² = AB²" pattern → right triangle at C
        if re.search(r'[A-Z][A-Z]\s*[²2]\s*\+\s*[A-Z][A-Z]\s*[²2]\s*=\s*[A-Z][A-Z]\s*[²2]', wm.raw_text):
            wm.add_geometry("right_triangle", [])

    # ── Distance extraction ──────────────────────────────────────────

    def _extract_point_distances(self, wm: WorldModel, text: str):
        """
        Extract explicit distances like 'AB = 4 m', 'AC = 12 cm',
        '8 cm apart', 'separated by 20 cm', 'distance of 5 m'.
        """
        # Named point-pair distances: AB = 4 m
        for m in re.finditer(r'\b([A-Z]{2})\s*=\s*([\d.]+)\s*(cm|mm|km|m)\b', text):
            pair, val_s, unit = m.group(1), m.group(2), m.group(3)
            val = float(val_s)
            if unit == "cm": val *= 1e-2
            elif unit == "mm": val *= 1e-3
            elif unit == "km": val *= 1e3
            wm.add_relation("distance", [pair[0], pair[1]],
                            Quantity(val, "m", "explicit"))

        # Narrative: "8 cm apart", "6 cm apart", "12 m apart"
        for m in re.finditer(
            r'([\d.]+)\s*(cm|mm|km|m)\s+apart\b',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            unit = m.group(2)
            if unit == "cm": val *= 1e-2
            elif unit == "mm": val *= 1e-3
            elif unit == "km": val *= 1e3
            if not any(r.relation_type == "distance" for r in wm.relations):
                wm.add_relation("distance", ["generic"], Quantity(val, "m", "explicit"))

        # "separated by X cm" / "distance of X m" / "r = X cm"
        for pattern in [
            r'separated\s+by\s+([\d.]+)\s*(cm|mm|km|m)\b',
            r'distance\s+(?:of\s+)?([\d.]+)\s*(cm|mm|km|m)\b',
            r'\br\s*=\s*([\d.]+)\s*(cm|mm|km|m)\b',
        ]:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = float(m.group(1))
                unit = m.group(2)
                if unit == "cm": val *= 1e-2
                elif unit == "mm": val *= 1e-3
                elif unit == "km": val *= 1e3
                if not any(r.relation_type == "distance" for r in wm.relations):
                    wm.add_relation("distance", ["generic"], Quantity(val, "m", "explicit"))

        # Generic r = X cm (as variable assignment)
        for m in re.finditer(
            r'\b(?:r|d)\s*=\s*([\d.]+)\s*(cm|mm|km|m)\b',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            unit = m.group(2)
            if unit == "cm": val *= 1e-2
            elif unit == "mm": val *= 1e-3
            elif unit == "km": val *= 1e3
            if not any(r.relation_type == "distance" for r in wm.relations):
                wm.add_relation("distance", ["generic"], Quantity(val, "m", "explicit"))


    # ── Derive missing side from Pythagoras ─────────────────────────

    def _derive_triangle_distances(self, wm: WorldModel):
        """
        If a right triangle is detected and exactly 2 sides are known,
        derive the third using Pythagoras.
        """
        if not wm.has_geometry("right_triangle"):
            return

        dist_rels = [r for r in wm.relations if r.relation_type == "distance" and r.value]
        if len(dist_rels) < 2:
            return

        sides = [r.value.value for r in dist_rels]
        # We know at most 3 sides; try to derive hypotenuse or leg
        if len(sides) == 2:
            hyp = math.sqrt(sides[0]**2 + sides[1]**2)
            wm.add_relation("distance", ["derived_hypotenuse"],
                            Quantity(hyp, "m", "inferred"))
        elif len(sides) == 3:
            pass  # All known

    # ── Angle inference ──────────────────────────────────────────────

    def _infer_angles_from_geometry(self, wm: WorldModel):
        """Set theta based on triangle type if not already present."""
        flat = wm.flat_quantities()
        if "theta" in flat:
            return   # already known

        if wm.has_geometry("equilateral_triangle"):
            wm.add_entity("angle", "geometry").set_attr(
                "theta", Quantity(math.radians(60), "rad", "inferred"))
        elif wm.has_geometry("right_triangle"):
            wm.add_entity("angle", "geometry").set_attr(
                "theta", Quantity(math.radians(90), "rad", "inferred"))
