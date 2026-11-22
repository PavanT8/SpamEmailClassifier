from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The email or SMS message content to classify."
    )

class EmailStats(BaseModel):
    char_count: int
    word_count: int
    tokens_analyzed: int

class PredictResponse(BaseModel):
    prediction: str = Field(..., description="'Spam' or 'Ham'")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")
    spam_score: int = Field(..., description="Spam likelihood score from 0 to 100")
    risk_level: str = Field(..., description="'Safe', 'Low Risk', 'Suspicious', or 'High Threat'")
    trigger_keywords: List[str] = Field(default_factory=list, description="Suspicious trigger words detected")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    stats: EmailStats = Field(..., description="Input length and token statistics")

class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'degraded'")
    model_loaded: bool
    model_name: str
    vocabulary_size: int
    uptime_seconds: int

class SampleEmail(BaseModel):
    id: str
    category: str
    title: str
    content: str
