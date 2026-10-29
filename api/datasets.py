"""CSV preview and schema validation endpoints."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, fields
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.prioritization.preprocessing.ratings import PTL_MAPPING
from src.prioritization.schemas import Claim, UserRating

PREVIEW_LIMIT = 5
MAX_ERRORS = 20

DATASET_FIELDS = {
    "claims": tuple(field.name for field in fields(Claim)),
    "perception": tuple(field.name for field in fields(UserRating)),
}
RATING_LABELS = ", ".join(PTL_MAPPING)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@dataclass(frozen=True, slots=True)
class ParsedCsv:
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def _resolve_dataset_type(dataset_type: str | None, request: Request) -> str:
    value = dataset_type
    if value is None:
        value = request.query_params.get("dataset_type")
    if value is None:
        raise HTTPException(
            status_code=422,
            detail="dataset_type is required: expected 'claims' or 'perception'.",
        )
    if value not in DATASET_FIELDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported dataset_type {value!r}: expected 'claims' or 'perception'.",
        )
    return value


async def read_upload(file: UploadFile) -> bytes:
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to read the uploaded CSV.") from exc
    if not content:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    return content


def _parse_csv(content: bytes) -> ParsedCsv:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        raw_header = next(reader, None)
        if raw_header is None:
            raise HTTPException(status_code=400, detail="CSV header row is missing.")

        columns = tuple(cell.strip() for cell in raw_header)
        if not columns or any(not column for column in columns):
            raise HTTPException(status_code=400, detail="CSV header contains an empty column name.")
        if len(set(columns)) != len(columns):
            raise HTTPException(status_code=400, detail="CSV header contains duplicate column names.")

        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            if len(row) > len(columns):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Malformed CSV at line {reader.line_num}: "
                        f"expected {len(columns)} fields, found {len(row)}."
                    ),
                )
            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
            rows.append(dict(zip(columns, row)))
    except csv.Error as exc:
        raise HTTPException(status_code=400, detail=f"Malformed CSV: {exc}") from exc

    return ParsedCsv(columns=columns, rows=tuple(rows))


def parse_dataset_csv(
    content: bytes,
    dataset_type: str,
) -> tuple[ParsedCsv, list[str], list[str]]:
    """Parse and validate one dataset CSV without raising schema errors."""
    parsed = _parse_csv(content)
    errors, warnings = _validate_rows(parsed, dataset_type)
    return parsed, errors, warnings


def _add_error(errors: list[str], message: str) -> None:
    if len(errors) < MAX_ERRORS:
        errors.append(message)
    elif len(errors) == MAX_ERRORS:
        errors.append(f"Additional errors omitted after {MAX_ERRORS} issues.")


def _validate_rows(parsed: ParsedCsv, dataset_type: str) -> tuple[list[str], list[str]]:
    required = DATASET_FIELDS[dataset_type]
    errors: list[str] = []
    warnings: list[str] = []

    missing = [column for column in required if column not in parsed.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}.")
    unexpected = [column for column in parsed.columns if column not in required]
    if unexpected:
        warnings.append(f"Unexpected column(s) will be ignored: {', '.join(unexpected)}.")
    if not parsed.rows:
        errors.append("CSV contains no data rows.")

    if missing:
        return errors, warnings

    seen_claim_ids: set[str] = set()
    for row_number, row in enumerate(parsed.rows, start=1):
        claim_id = row["claim_id"]
        if not claim_id.strip():
            _add_error(errors, f"Row {row_number}: claim_id must not be empty.")
        elif dataset_type == "claims":
            if claim_id in seen_claim_ids:
                _add_error(errors, f"Row {row_number}: duplicate claim_id {claim_id!r}.")
            else:
                seen_claim_ids.add(claim_id)

        if dataset_type == "claims":
            if not row["claim"].strip():
                _add_error(errors, f"Row {row_number}: claim text must not be empty.")
        else:
            rating = row["rating"]
            if rating not in PTL_MAPPING:
                _add_error(
                    errors,
                    f"Row {row_number}: unsupported rating {rating!r}; expected one of {RATING_LABELS}.",
                )

    return errors, warnings


@router.post("/preview")
async def preview_dataset(
    request: Request,
    file: UploadFile = File(...),
    dataset_type: str | None = Form(None),
) -> dict[str, Any]:
    """Return the CSV columns, row count, and up to five preview rows."""
    resolved_type = _resolve_dataset_type(dataset_type, request)
    parsed = _parse_csv(await read_upload(file))
    return {
        "dataset_type": resolved_type,
        "columns": list(parsed.columns),
        "row_count": len(parsed.rows),
        "preview": list(parsed.rows[:PREVIEW_LIMIT]),
    }


@router.post("/validate")
async def validate_dataset(
    request: Request,
    file: UploadFile = File(...),
    dataset_type: str | None = Form(None),
) -> dict[str, Any]:
    """Validate a CSV against the claims or perception dataset schema."""
    resolved_type = _resolve_dataset_type(dataset_type, request)
    parsed, errors, warnings = parse_dataset_csv(
        await read_upload(file),
        resolved_type,
    )
    return {
        "dataset_type": resolved_type,
        "valid": not errors,
        "row_count": len(parsed.rows),
        "columns": list(parsed.columns),
        "errors": errors,
        "warnings": warnings,
    }
