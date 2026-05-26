from enum import Enum
from pydantic import BaseModel


class RoutingStrategy(str, Enum):
    STATIC = "static"
    LEADERBOARD = "leaderboard"
    ADAPTIVE = "adaptive"
    THOMPSON = "thompson"


class Message(BaseModel):
    role: str
    content: str


class RouteRequest(BaseModel):
    messages: list[Message]
    preferred_labels: list[str] = []
    strategy: RoutingStrategy = RoutingStrategy.LEADERBOARD
    enable_realtime_eval: bool | None = None
