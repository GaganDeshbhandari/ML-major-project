"""Tests for the CSV preview and validation endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

PREVIEW_URL = "/api/v1/datasets/preview"
VALIDATE_URL = "/api/v1/datasets/validate"


def _post_csv(url: str, content: bytes, dataset_type: str = "claims"):
    return client.post(
        url,
        files={"file": ("dataset.csv", content, "text/csv")},
        data={"dataset_type": dataset_type},
    )


def test_preview_returns_columns_row_count_and_at_most_five_rows() -> None:
    data_rows = "\n".join(f"{index},Claim {index}" for index in range(1, 8))
    response = _post_csv(PREVIEW_URL, f"claim_id,claim\n{data_rows}\n".encode())

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_type"] == "claims"
    assert body["columns"] == ["claim_id", "claim"]
    assert body["row_count"] == 7
    assert len(body["preview"]) == 5
    assert body["preview"][0] == {"claim_id": "1", "claim": "Claim 1"}


def test_validate_accepts_valid_claims_csv() -> None:
    response = _post_csv(VALIDATE_URL, b"claim_id,claim\n1,First claim\n2,Second claim\n")

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["row_count"] == 2
    assert body["errors"] == []


def test_validate_accepts_valid_perception_csv() -> None:
    content = b"claim_id,rating\n1,True\n1,Mostly False\n2,Mixture\n"
    response = _post_csv(VALIDATE_URL, content, dataset_type="perception")

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["dataset_type"] == "perception"


@pytest.mark.parametrize("url", [PREVIEW_URL, VALIDATE_URL])
def test_invalid_dataset_type_is_rejected(url: str) -> None:
    response = _post_csv(url, b"claim_id,claim\n1,Example\n", dataset_type="news")

    assert response.status_code == 422
    assert "dataset_type" in response.json()["detail"]


@pytest.mark.parametrize("url", [PREVIEW_URL, VALIDATE_URL])
def test_empty_csv_is_rejected(url: str) -> None:
    response = _post_csv(url, b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "CSV file is empty."


@pytest.mark.parametrize(
    "content",
    [
        b'claim_id,claim\n1,"unterminated\n',
        b"claim_id,claim\n1,Example,extra\n",
    ],
)
def test_malformed_csv_is_rejected(content: bytes) -> None:
    response = _post_csv(VALIDATE_URL, content)

    assert response.status_code == 400
    assert "Malformed CSV" in response.json()["detail"]


@pytest.mark.parametrize(
    ("dataset_type", "content", "missing_column"),
    [
        ("claims", b"claim_id\n1\n", "claim"),
        ("perception", b"claim_id\n1\n", "rating"),
    ],
)
def test_missing_required_columns_are_reported(
    dataset_type: str,
    content: bytes,
    missing_column: str,
) -> None:
    response = _post_csv(VALIDATE_URL, content, dataset_type=dataset_type)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert any(missing_column in error for error in body["errors"])


def test_invalid_perception_rating_is_reported() -> None:
    response = _post_csv(
        VALIDATE_URL,
        b"claim_id,rating\n1,Very True\n",
        dataset_type="perception",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert any("unsupported rating" in error for error in body["errors"])
