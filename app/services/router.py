from app.models.ai_model import AIModel


def select_model(models: list[AIModel], preferred_labels: list[str]) -> AIModel | None:
    if not models:
        return None

    if preferred_labels:
        matched = [
            m for m in models if set(preferred_labels) & set(m.labels)
        ]
        if matched:
            return min(matched, key=lambda m: m.cost_per_1k_tokens)

    return min(models, key=lambda m: m.cost_per_1k_tokens)
