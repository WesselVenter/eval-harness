import os

import pytest

from eval.metrics import check_thresholds
from eval.run_eval import run_eval

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


def test_eval_meets_thresholds():
    summary, _ = run_eval()
    violations = check_thresholds(summary)
    assert not violations, f"Threshold violations: {violations}"
