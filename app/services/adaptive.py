import random
import asyncio
import time
from dataclasses import dataclass, field
from collections import defaultdict

import httpx

from app.config import settings
from app.models.ai_model import AIModel


@dataclass
class EvalResult:
    model: AIModel
    snippet: str
    relevance_score: float
    fluency_score: float
    combined_score: float
    cost_adjusted_score: float


@dataclass
class ModelHistory:
    alpha: float = 1.0  # successes + 1 (Beta distribution param)
    beta: float = 1.0   # failures + 1 (Beta distribution param)
    total_score: float = 0.0
    count: int = 0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.count if self.count > 0 else 0.5

    def update(self, score: float):
        self.total_score += score
        self.count += 1
        self.alpha += score
        self.beta += (1.0 - score)

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)


# user_id -> model_name -> label_key -> ModelHistory
_history: dict[int, dict[str, dict[str, ModelHistory]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(ModelHistory))
)


def get_history(user_id: int, model_name: str, label_key: str) -> ModelHistory:
    return _history[user_id][model_name][label_key]


def _label_key(preferred_labels: list[str]) -> str:
    return "|".join(sorted(preferred_labels)) if preferred_labels else "__general__"


async def generate_snippet(
    api_url: str, api_key: str, model_name: str, messages: list[dict]
) -> tuple[str, dict | None]:
    try:
        async with httpx.AsyncClient(timeout=settings.EVAL_TIMEOUT) as client:
            resp = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": settings.EVAL_SNIPPET_MAX_TOKENS,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")
            return content, data.get("usage")
    except Exception:
        return "", None


async def evaluate_snippet(snippet: str, user_prompt: str) -> tuple[float, float]:
    if not settings.EVAL_MODEL_URL or not snippet:
        return 0.0, 0.0

    eval_prompt = (
        f"Rate the following AI response snippet on two dimensions.\n"
        f"User question: {user_prompt}\n"
        f"Response snippet: {snippet}\n\n"
        f"Score each dimension from 0 to 1 (decimals allowed):\n"
        f"1. Relevance: How relevant is the response to the user's question?\n"
        f"2. Fluency: How natural and well-written is the response?\n\n"
        f"Reply ONLY in this exact format:\n"
        f"relevance=<score>\nfluency=<score>"
    )

    try:
        async with httpx.AsyncClient(timeout=settings.EVAL_TIMEOUT) as client:
            resp = await client.post(
                settings.EVAL_MODEL_URL,
                headers={
                    "Authorization": f"Bearer {settings.EVAL_MODEL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.EVAL_MODEL_NAME,
                    "messages": [{"role": "user", "content": eval_prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            return _parse_eval_scores(text)
    except Exception:
        return 0.5, 0.5


def _parse_eval_scores(text: str) -> tuple[float, float]:
    relevance, fluency = 0.5, 0.5
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line.startswith("relevance"):
            try:
                relevance = float(line.split("=")[1].strip())
                relevance = max(0.0, min(1.0, relevance))
            except (IndexError, ValueError):
                pass
        elif line.startswith("fluency"):
            try:
                fluency = float(line.split("=")[1].strip())
                fluency = max(0.0, min(1.0, fluency))
            except (IndexError, ValueError):
                pass
    return relevance, fluency


async def adaptive_select(
    models: list[AIModel],
    messages: list[dict],
    preferred_labels: list[str],
    user_id: int,
) -> tuple[AIModel | None, list[EvalResult], float]:
    if not models:
        return None, [], 0.0

    candidate_count = min(settings.EVAL_CANDIDATE_COUNT, len(models))
    candidates = random.sample(models, candidate_count)

    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    start = time.perf_counter()

    snippet_tasks = [
        generate_snippet(m.api_url, m.api_key, m.name, messages)
        for m in candidates
    ]
    snippets = await asyncio.gather(*snippet_tasks)

    eval_tasks = [
        evaluate_snippet(snippet, user_prompt)
        for snippet, _ in snippets
    ]
    eval_scores = await asyncio.gather(*eval_tasks)

    eval_duration = time.perf_counter() - start

    label_key = _label_key(preferred_labels)
    results: list[EvalResult] = []

    for i, model in enumerate(candidates):
        snippet_text, _ = snippets[i]
        relevance, fluency = eval_scores[i]
        combined = (relevance + fluency) / 2.0

        history = get_history(user_id, model.name, label_key)
        history_weight = settings.EVAL_HISTORY_WEIGHT
        weighted_score = (1 - history_weight) * combined + history_weight * history.avg_score

        cost = model.cost_per_1k_tokens if model.cost_per_1k_tokens > 0 else 0.001
        cost_adjusted = weighted_score / cost

        results.append(EvalResult(
            model=model,
            snippet=snippet_text,
            relevance_score=relevance,
            fluency_score=fluency,
            combined_score=combined,
            cost_adjusted_score=cost_adjusted,
        ))

    if not results:
        return None, [], eval_duration

    best = max(results, key=lambda r: r.cost_adjusted_score)

    history = get_history(user_id, best.model.name, label_key)
    history.update(best.combined_score)

    return best.model, results, eval_duration


def thompson_select(
    models: list[AIModel],
    preferred_labels: list[str],
    user_id: int,
) -> tuple[AIModel | None, float]:
    if not models:
        return None, 0.0

    label_key = _label_key(preferred_labels)
    best_model = None
    best_score = -1.0

    for model in models:
        history = get_history(user_id, model.name, label_key)
        sampled = history.sample()
        cost = model.cost_per_1k_tokens if model.cost_per_1k_tokens > 0 else 0.001
        score = sampled / cost
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score
