# src/reasoning/geometry_helpers.py

import math
from typing import Dict, Any

class GeometryHelper:
    """
    Infers missing geometric variables (like angles) from natural language keywords.
    """
    
    @staticmethod
    def infer_angles(text: str, variables: Dict[str, float]) -> Dict[str, float]:
        """
        Detect keywords and inject angles (in radians) into the known variables.
        """
        text_lower = text.lower()
        
        # Right triangle implies 90 degrees (pi/2)
        if "right triangle" in text_lower or "orthogonal" in text_lower or "perpendicular" in text_lower:
            variables["theta"] = math.pi / 2
            
        # Equilateral triangle implies 60 degrees (pi/3)
        if "equilateral" in text_lower:
            variables["theta"] = math.pi / 3
            
        # Parallel implies 0 degrees
        if "parallel" in text_lower and "anti-parallel" not in text_lower:
            variables["theta"] = 0
            
        # Opposite direction implies 180 degrees
        if "opposite direction" in text_lower or "anti-parallel" in text_lower:
            variables["theta"] = math.pi
            
        return variables
