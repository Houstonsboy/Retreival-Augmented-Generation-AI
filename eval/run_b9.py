"""B9 test harness: runs s106b_test_bundles.json through the gate and prints results."""

import json
import sys
import types
from pathlib import Path

# SCCVision/__init__.py imports cv2 (not available here).
# Stub it out before the real package loads so the s106b subpackage is importable.
_sccvision_stub = types.ModuleType("SCCVision")
_sccvision_stub.__path__ = [str(Path(__file__).parent.parent / "backend" / "SCCVision")]
_sccvision_stub.__package__ = "SCCVision"
_sccvision_stub.__spec__ = None
sys.modules.setdefault("SCCVision", _sccvision_stub)

from SCCVision.s106b.gate import evaluate   # noqa: E402  (must come after stub)
from SCCVision.s106b.models import DeviceFacts  # noqa: E402

BUNDLES_PATH = Path(__file__).parent / "s106b_test_bundles.json"


def main():
    bundles = json.loads(BUNDLES_PATH.read_text())

    # Column widths
    col = {"id": 8, "expected": 22, "actual": 22, "unmet": 30, "match": 5}
    header = (
        f"{'Bundle':<{col['id']}} | "
        f"{'Expected':<{col['expected']}} | "
        f"{'Actual':<{col['actual']}} | "
        f"{'Failing conditions':<{col['unmet']}} | "
        f"Match"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    all_pass = True

    for bundle in bundles:
        bid          = bundle["bundle_id"]
        snapshot     = bundle["analysis_snapshot"]
        expected_out = bundle["expected_outcome"]
        expected_unmet = sorted(bundle.get("expected_unmet_conditions", []))

        facts = DeviceFacts(**bundle["device_facts"])
        result = evaluate(snapshot, facts)

        actual_out   = result.outcome
        actual_unmet = sorted(result.unmet)

        match = (actual_out == expected_out) and (actual_unmet == expected_unmet)
        if not match:
            all_pass = False

        failing_str = ", ".join(actual_unmet) if actual_unmet else "(none)"

        print(
            f"{bid:<{col['id']}} | "
            f"{expected_out:<{col['expected']}} | "
            f"{actual_out:<{col['actual']}} | "
            f"{failing_str:<{col['unmet']}} | "
            f"{'Y' if match else 'N'}"
        )

    print(separator)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
