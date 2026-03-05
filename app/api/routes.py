from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    WasteRequest,
    WasteClassificationResponse,
    WasteDisposalResponse,
)
from app.services.waste_audit_service import WasteAuditService

router = APIRouter(tags=["Waste Auditor"])

waste_audit_service = WasteAuditService()


@router.get("/health", summary="Health check endpoint")
def health_check():
    """Check if API is running."""
    return {"status": "healthy"}


@router.post(
    "/classify",
    response_model=WasteClassificationResponse,
    summary="Classify waste item",
    description="Classifies waste text into a category using the ML model."
)
def classify_waste(request: WasteRequest):
    """
    ML-only waste classification.
    No LLM call.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = waste_audit_service.classify(request.text)

    return WasteClassificationResponse(
        text=request.text,
        category=result["label"],
        confidence=result["confidence"],
    )


@router.post(
    "/disposal",
    response_model=WasteDisposalResponse,
    summary="Generate disposal plan",
    description="Uses ML + city policy + LLM to generate safe disposal instructions."
)
def disposal_plan(request: WasteRequest):
    """
    Full pipeline: ML classification + policy + LLM reasoning.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = waste_audit_service.generate_disposal_plan(request.text)

    return WasteDisposalResponse(**result)
