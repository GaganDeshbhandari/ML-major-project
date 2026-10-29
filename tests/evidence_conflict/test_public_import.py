"""Tests for the public evidence conflict package surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import src.evidence_conflict as evidence_conflict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = {
    "DEFAULT_NLI_MODEL",
    "ECSConfig",
    "ECSConfigurationError",
    "ECSError",
    "ECSStatus",
    "EvidenceConflictError",
    "EvidenceConflictResult",
    "StanceCounts",
    "StanceInferenceError",
    "StanceInputError",
    "StanceLabel",
    "StanceModelError",
    "StancePrediction",
    "StanceProbabilities",
    "UnsupportedLabelMappingError",
}


def test_public_exports_are_stable() -> None:
    assert set(evidence_conflict.__all__) == EXPECTED_EXPORTS
    for name in EXPECTED_EXPORTS:
        assert hasattr(evidence_conflict, name)


def test_public_import_does_not_load_heavy_backends() -> None:
    code = "\n".join(
        [
            "import os",
            "import sys",
            "os.environ['ECS_BATCH_SIZE'] = '99'",
            "import src.evidence_conflict as ec",
            "assert ec.ECSConfig().batch_size == 8, "
            "'ECSConfig read the environment at import time'",
            "assert 'torch' not in sys.modules, 'torch was imported'",
            "assert 'transformers' not in sys.modules, "
            "'transformers was imported'",
            "assert 'sentence_transformers' not in sys.modules, "
            "'sentence_transformers was imported'",
            "print('ok')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
