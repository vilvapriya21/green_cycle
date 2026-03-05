"""
API routes for the Green-Cycle Waste Auditor.

Exposes:
- GET  /health   : Health check
- POST /classify : Waste category prediction
- POST /disposal : Waste category + disposal plan generation
"""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.models import (
    WasteRequest,
    WasteClassificationResponse,
    WasteDisposalResponse,
)
from app.services.waste_audit_service import WasteAuditService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Waste Auditor"])


@lru_cache
def get_waste_audit_service() -> WasteAuditService:
    """
    Provide a cached instance of WasteAuditService.

    Using lru_cache ensures the service (and its ML model) is initialized once
    per process rather than on every request.
    """
    return WasteAuditService()


@router.get(
    "/health",
    summary="Health check endpoint",
    response_model=dict,
)
def health_check() -> dict:
    """
    Health check endpoint to verify that the API is running.

    Returns:
        dict: A simple status payload.
    """
    return {"status": "healthy"}


@router.post(
    "/classify",
    response_model=WasteClassificationResponse,
    summary="Classify waste item",
    description="Classifies waste text into a category using the ML model.",
)
def classify_waste(
    request: WasteRequest,
    svc: WasteAuditService = Depends(get_waste_audit_service),
) -> WasteClassificationResponse:
    """
    Classify a waste description into a waste category.

    Args:
        request (WasteRequest): Input payload containing waste description text.
        svc (WasteAuditService): Service dependency.

    Returns:
        WasteClassificationResponse: Predicted category and confidence score.
    """
    raw = svc.classify(request.text)

    # svc.classify() returns {"label": "...", "confidence": ...}
    result: WasteClassificationResponse = {
        "text": request.text,
        "category": raw.get("label", "Uncertain"),
        "confidence": float(raw.get("confidence", 0.0)),
    }

    logger.info(
        "POST /classify | text_len=%d | category=%s | confidence=%.3f",
        len(request.text),
        result["category"],
        result["confidence"],
    )

    return result


@router.post(
    "/disposal",
    response_model=WasteDisposalResponse,
    summary="Generate disposal plan",
    description="Uses ML + city policy + LLM to generate safe disposal instructions.",
)
def disposal_plan(
    request: WasteRequest,
    svc: WasteAuditService = Depends(get_waste_audit_service),
) -> WasteDisposalResponse:
    """
    Generate a disposal plan for a waste description.

    The response includes:
    - predicted category
    - confidence score
    - disposal plan generated using policy rules and an LLM (with safe fallback)

    Args:
        request (WasteRequest): Input payload containing waste description text.
        svc (WasteAuditService): Service dependency.

    Returns:
        WasteDisposalResponse: Category, confidence, and disposal instructions.
    """
    result = svc.generate_disposal_plan(request.text)

    # Defensive programming: never return None (prevents FastAPI response validation errors)
    if result is None:
        logger.error("POST /disposal | service returned None")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate disposal plan.",
        )

    logger.info(
        "POST /disposal | text_len=%d | category=%s | confidence=%.3f",
        len(request.text),
        result.get("category", "Uncertain"),
        float(result.get("confidence", 0.0)),
    )

    return result
