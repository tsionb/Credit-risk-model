from pydantic import BaseModel
from typing import List

class CreditRiskInput(BaseModel):
    features: List[float]
