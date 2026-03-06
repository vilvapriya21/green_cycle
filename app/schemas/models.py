from pydantic import BaseModel, Field


class WasteRequest(BaseModel):
    """
    Request model for waste classification and disposal planning.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text description of the waste item",
        json_schema_extra={"example": "banana peel"}
    )


class WasteClassificationResponse(BaseModel):
    """
    Response model for classification endpoint.
    """

    text: str
    category: str
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model confidence score"
    )


class WasteDisposalResponse(BaseModel):
    """
    Response model for disposal recommendation.
    """

    text: str
    category: str
    confidence: float = Field(
        ...,
        ge=0,
        le=1
    )

    disposal_plan: str = Field(
        ...,
        description="AI-generated safe disposal instructions"
    )
