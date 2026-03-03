from pydantic import BaseModel, Field


class WasteRequest(BaseModel):
    """
    Request model for waste classification and disposal planning.
    """
    text: str = Field(
        ...,
        min_length=1,
        description="Text description of waste item"
    )


class WasteClassificationResponse(BaseModel):
    """
    Response model for classification endpoint.
    Returns only ML prediction.
    """
    text: str
    category: str
    confidence: float


class WasteDisposalResponse(BaseModel):
    """
    Response model for disposal endpoint.
    Returns classification + AI-generated disposal plan.
    """
    text: str
    category: str
    confidence: float
    disposal_plan: str