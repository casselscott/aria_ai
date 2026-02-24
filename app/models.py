# models.py - Data models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Property(BaseModel):
    id: str
    title: str
    area: str
    price: str
    price_value: int  # Numeric value for sorting
    bedrooms: int
    bathrooms: int
    sqft: str
    type: str
    features: List[str]
    description: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    category: str = "premium"  # premium or luxury
    status: str = "For Sale"
    year: Optional[int] = None
    url: Optional[str] = None

class ChatRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    success: bool = True
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[Property]
    count: int
    success: bool = True
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    properties_count: int
    environment: str

class StatsResponse(BaseModel):
    total_properties: int
    areas: Dict[str, int]
    property_types: Dict[str, int]
    bedroom_counts: Dict[int, int]
    price_range: Dict[str, str]
    last_updated: datetime = Field(default_factory=datetime.now)