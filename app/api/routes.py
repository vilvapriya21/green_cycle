from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    WasteRequest,
    WasteClassificationResponse,
    WasteDisposalResponse,
)
from app.services.waste_audit_service import WasteAuditService

router = APIRouter()
waste_audit_service = WasteAuditService()


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.post("/classify", response_model=WasteClassificationResponse)
def classify_waste(request: WasteRequest):
    """
    Classify a waste item description into a waste category.
    ML only — no LLM call.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = waste_audit_service.classify(request.text)

        return WasteClassificationResponse(
            text=request.text,
            category=result["label"],
            confidence=result["confidence"],
        )

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/disposal", response_model=WasteDisposalResponse)
def disposal_plan(request: WasteRequest):
    """
    Generate full disposal plan using ML + policy + LLM.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = waste_audit_service.generate_disposal_plan(request.text)
        return WasteDisposalResponse(**result)

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")
