from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from bson import ObjectId

# Custom class to handle ObjectId in Pydantic models
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

# Item model matching your MongoDB schema
class ItemModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(...)
    rating: float = Field(default=0, ge=0, le=5)
    price: float = Field(..., description="Item price is required")
    description: str = Field(..., description="Item description is required")
    colors: List[str] = Field(default_factory=list)
    images: List[str] = Field(...)
    reviewCount: int = Field(default=0)
    img: Optional[str] = Field(default=None)
    gender: Literal["male", "female", "neutral"] = Field(default="male")
    category: PyObjectId = Field(...)
    categoryField: Optional[str] = None
    seller: PyObjectId = Field(...)
    sizes: List[str] = Field(default_factory=list)
    brand: Optional[PyObjectId] = None
    material: Optional[str] = None
    countryOfOrigin: Optional[str] = None
    featured: bool = Field(default=False)
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }

# Response model for similar items
class SimilarItemsResponse(BaseModel):
    similar_items: List[ItemModel]
