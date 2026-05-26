from app.models.ai_model import AIModel
from app.services.leaderboard import get_cached_benchmarks

LABEL_TO_DOMAIN = {
    "math": "math",
    "数学": "math",
    "reasoning": "reasoning",
    "推理": "reasoning",
    "knowledge": "knowledge",
    "知识": "knowledge",
    "science": "science",
    "科学": "science",
    "truthfulness": "truthfulness",
    "instruction_following": "instruction_following",
    "code": "code",
    "编程": "code",
    "creative": "creative",
    "创意": "creative",
}


def _find_benchmark(model_name: str, benchmarks: list[dict]) -> dict | None:
    name_lower = model_name.lower().strip()
    for b in benchmarks:
        if b["model_name"].lower().strip() == name_lower:
            return b
    for b in benchmarks:
        if name_lower in b["model_name"].lower() or b["model_name"].lower() in name_lower:
            return b
    return None


def _get_domain_score(benchmark: dict, preferred_labels: list[str]) -> float | None:
    domain_scores = benchmark.get("domain_scores", {})
    if not domain_scores or not preferred_labels:
        return None

    matched_scores = []
    for label in preferred_labels:
        domain = LABEL_TO_DOMAIN.get(label.lower())
        if domain and domain in domain_scores:
            matched_scores.append(domain_scores[domain])

    return max(matched_scores) if matched_scores else None


def select_model(
    models: list[AIModel],
    preferred_labels: list[str],
) -> tuple[AIModel | None, float | None]:
    if not models:
        return None, None

    if preferred_labels:
        matched = [
            m for m in models if set(preferred_labels) & set(m.labels)
        ]
        candidates = matched if matched else models
    else:
        candidates = models

    benchmarks = get_cached_benchmarks()
    if not benchmarks:
        chosen = min(candidates, key=lambda m: m.cost_per_1k_tokens)
        return chosen, None

    scored = []
    for model in candidates:
        benchmark = _find_benchmark(model.name, benchmarks)
        if not benchmark:
            continue

        if preferred_labels:
            domain_score = _get_domain_score(benchmark, preferred_labels)
            if domain_score is None:
                continue
            score = domain_score
        else:
            score = benchmark["overall_score"]

        cost = model.cost_per_1k_tokens
        if cost <= 0:
            cost = 0.001
        value_score = (score / 100.0) / cost
        scored.append((model, value_score, score))

    if not scored:
        chosen = min(candidates, key=lambda m: m.cost_per_1k_tokens)
        return chosen, None

    best = max(scored, key=lambda x: x[1])
    return best[0], best[2]
