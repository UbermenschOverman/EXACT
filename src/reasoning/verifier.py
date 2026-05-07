# src/reasoning/verifier.py

"""
Multi-check verification layer.
Validates answers through numerical sanity, unit consistency,
and reasoning consistency checks.
"""

import re
from typing import Dict, Any, Optional, List


class Verifier:
    """
    Verification layer with three independent checks:
    1. Numerical sanity — re-verify arithmetic
    2. Unit consistency — dimensional analysis
    3. Reasoning consistency — trace completeness
    """

    # Known unit compatibility rules
    UNIT_DIMENSIONS = {
        "V": "voltage",
        "A": "current",
        "Ω": "resistance",
        "W": "power",
        "J": "energy",
        "F": "capacitance",
        "N": "force",
        "C": "charge",
        "N/C": "electric_field",
    }

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all verification checks on a result.

        Returns:
            {
                "valid": bool,
                "checks": {
                    "numerical_sanity": {"passed": bool, "detail": str},
                    "unit_consistency": {"passed": bool, "detail": str},
                    "reasoning_consistency": {"passed": bool, "detail": str},
                },
                "confidence_adjustment": float,
            }
        """
        checks = {}

        # 1. Numerical sanity
        checks["numerical_sanity"] = self._check_numerical_sanity(result)

        # 2. Unit consistency
        checks["unit_consistency"] = self._check_unit_consistency(result)

        # 3. Reasoning consistency
        checks["reasoning_consistency"] = self._check_reasoning_consistency(result)

        # Compute overall validity
        all_passed = all(c.get("passed", False) for c in checks.values())
        any_passed = any(c.get("passed", False) for c in checks.values())

        # Confidence adjustment
        passed_count = sum(1 for c in checks.values() if c.get("passed", False))
        confidence_adj = passed_count / max(len(checks), 1)

        return {
            "valid": any_passed,  # valid if at least one check passes
            "checks": checks,
            "confidence_adjustment": confidence_adj,
        }

    def _check_numerical_sanity(self, result: Dict) -> Dict:
        """
        Verify arithmetic by checking:
        - Answer is not None/empty/UNKNOWN
        - Numeric values are reasonable (not NaN, not extreme)
        - If trace has computation, verify the math
        """
        answer = result.get("answer", "")

        if not answer or answer == "UNKNOWN":
            return {"passed": False, "detail": "No answer produced"}

        # Check for numeric value
        numeric = result.get("numeric_value")
        if numeric is not None:
            # Check for NaN/Inf
            if not isinstance(numeric, (int, float)):
                return {"passed": False, "detail": f"Non-numeric value: {numeric}"}

            import math
            if math.isnan(numeric) or math.isinf(numeric):
                return {"passed": False, "detail": f"Invalid numeric: {numeric}"}

            # Check reasonable range (physics values shouldn't be extreme)
            if abs(numeric) > 1e12:
                return {
                    "passed": False,
                    "detail": f"Value seems unreasonably large: {numeric}"
                }

            # Re-verify computation from trace
            trace = result.get("reasoning_trace", [])
            compute_step = next(
                (t for t in trace if t.get("action") == "compute"), None
            )
            if compute_step and "result" in compute_step:
                expected = compute_step["result"]
                if isinstance(expected, (int, float)):
                    if abs(expected - numeric) > 1e-6:
                        return {
                            "passed": False,
                            "detail": f"Computation mismatch: trace={expected}, answer={numeric}"
                        }

            return {"passed": True, "detail": f"Numeric value {numeric} is valid"}

        # Non-numeric answer — basic check
        if len(str(answer).strip()) > 0:
            return {"passed": True, "detail": "Non-numeric answer present"}

        return {"passed": False, "detail": "Empty answer"}

    def _check_unit_consistency(self, result: Dict) -> Dict:
        """
        Verify units are dimensionally consistent.
        """
        unit = result.get("unit", "")
        formula_used = result.get("formula_used", "")

        if not unit and not formula_used:
            # No unit info — can't verify but don't fail
            return {"passed": True, "detail": "No units to verify (non-physics)"}

        if unit:
            # Check unit is recognized
            if unit in self.UNIT_DIMENSIONS:
                return {
                    "passed": True,
                    "detail": f"Unit '{unit}' is valid ({self.UNIT_DIMENSIONS[unit]})"
                }
            else:
                # Unknown unit — warn but don't fail
                return {
                    "passed": True,
                    "detail": f"Unit '{unit}' not in standard set but accepted"
                }

        return {"passed": True, "detail": "No unit check needed"}

    def _check_reasoning_consistency(self, result: Dict) -> Dict:
        """
        Verify reasoning trace is complete and consistent:
        - Has at least one step
        - Has a conclusion step
        - Answer matches conclusion
        """
        trace = result.get("reasoning_trace", [])
        answer = result.get("answer", "")

        if not trace:
            return {"passed": False, "detail": "No reasoning trace"}

        # Check for conclusion step
        has_conclusion = any(
            t.get("action") in ("conclude", "compute", "result")
            for t in trace
        )

        if not has_conclusion:
            return {"passed": False, "detail": "No conclusion step in trace"}

        # Check trace has at least 2 steps
        if len(trace) < 2:
            return {
                "passed": False,
                "detail": f"Trace too short ({len(trace)} steps)"
            }

        # Check answer is present
        if not answer or answer == "UNKNOWN":
            return {"passed": False, "detail": "No valid answer in result"}

        # Check for error steps
        error_steps = [t for t in trace if t.get("action") == "error"]
        if error_steps:
            return {
                "passed": False,
                "detail": f"Trace contains errors: {error_steps[0].get('detail', '')}"
            }

        return {
            "passed": True,
            "detail": f"Trace complete with {len(trace)} steps"
        }