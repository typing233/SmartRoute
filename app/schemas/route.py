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


class EvalCandidateInfo(BaseModel):
    model_name: str
    snippet: str
    relevance_score: float
    fluency_score: float
    combined_score: float
    cost_adjusted_score: float


class AdaptiveRouteMetadata(BaseModel):
    strategy_used: str
    selected_model: str
    eval_duration_ms: float
    candidates_evaluated: list[EvalCandidateInfo] = []
