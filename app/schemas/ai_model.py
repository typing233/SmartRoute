from pydantic import BaseModel


class AIModelCreate(BaseModel):
    name: str
    api_url: str
    api_key: str
    cost_per_1k_tokens: float
    labels: list[str] = []


class AIModelOut(BaseModel):
    id: int
    name: str
    api_url: str
    cost_per_1k_tokens: float
    labels: list[str]

    model_config = {"from_attributes": True}
