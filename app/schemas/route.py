from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class RouteRequest(BaseModel):
    messages: list[Message]
    preferred_labels: list[str] = []
