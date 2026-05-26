import logging
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.database import async_session
from app.models.benchmark import ModelBenchmark

logger = logging.getLogger(__name__)

DOMAIN_MAPPING = {
    "gsm8k": "math",
    "math": "math",
    "mmlu": "knowledge",
    "hellaswag": "reasoning",
    "arc_challenge": "reasoning",
    "winogrande": "reasoning",
    "truthfulqa": "truthfulness",
    "ifeval": "instruction_following",
    "bbh": "reasoning",
    "musr": "reasoning",
    "gpqa": "science",
    "math_lvl5": "math",
}

_cache: list[dict] = []
_cache_time: datetime | None = None


def get_cached_benchmarks() -> list[dict]:
    return _cache


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
async def _fetch_from_huggingface() -> list[dict]:
    url = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/models"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _parse_huggingface_data(raw_data: list[dict]) -> list[dict]:
    results = []
    for entry in raw_data:
        model_name = entry.get("model", {}).get("name", "") if isinstance(entry.get("model"), dict) else entry.get("model_name", entry.get("Model", ""))
        if not model_name:
            continue

        scores = {}
        overall = None

        if isinstance(entry.get("results"), dict):
            for benchmark_key, score_val in entry["results"].items():
                key_lower = benchmark_key.lower().replace(" ", "_")
                if isinstance(score_val, (int, float)):
                    scores[key_lower] = score_val
                elif isinstance(score_val, dict):
                    scores[key_lower] = score_val.get("score", score_val.get("value", 0))
            if scores:
                overall = sum(scores.values()) / len(scores)

        if overall is None:
            overall = entry.get("average", entry.get("Average", entry.get("score", 0)))
            if isinstance(overall, str):
                try:
                    overall = float(overall)
                except ValueError:
                    overall = 0

        if not overall:
            continue

        domain_scores = {}
        for bench_key, score in scores.items():
            domain = DOMAIN_MAPPING.get(bench_key)
            if domain:
                domain_scores[domain] = max(domain_scores.get(domain, 0), score)

        results.append({
            "source": "huggingface_open_llm",
            "model_name": model_name,
            "overall_score": round(float(overall), 2),
            "domain_scores": domain_scores,
        })

    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
async def _fetch_from_lmsys() -> list[dict]:
    url = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/results.json"
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _parse_lmsys_data(raw_data) -> list[dict]:
    results = []
    items = raw_data if isinstance(raw_data, list) else raw_data.get("data", raw_data.get("rows", []))

    for entry in items:
        if isinstance(entry, dict):
            model_name = entry.get("model", entry.get("Model", entry.get("name", "")))
            elo = entry.get("elo", entry.get("rating", entry.get("Arena Elo", 0)))
        elif isinstance(entry, list) and len(entry) >= 2:
            model_name = entry[0]
            elo = entry[1]
        else:
            continue

        if not model_name:
            continue

        try:
            elo_float = float(elo)
        except (ValueError, TypeError):
            continue

        score = round(min(max((elo_float - 800) / 4, 0), 100), 2)

        results.append({
            "source": "lmsys_chatbot_arena",
            "model_name": model_name,
            "overall_score": score,
            "domain_scores": {},
        })

    return results


async def fetch_and_store_benchmarks():
    global _cache, _cache_time

    all_benchmarks = []

    try:
        hf_raw = await _fetch_from_huggingface()
        hf_parsed = _parse_huggingface_data(hf_raw)
        all_benchmarks.extend(hf_parsed)
        logger.info(f"Fetched {len(hf_parsed)} models from HuggingFace Open LLM Leaderboard")
    except Exception as e:
        logger.warning(f"Failed to fetch HuggingFace leaderboard: {e}")

    try:
        lmsys_raw = await _fetch_from_lmsys()
        lmsys_parsed = _parse_lmsys_data(lmsys_raw)
        all_benchmarks.extend(lmsys_parsed)
        logger.info(f"Fetched {len(lmsys_parsed)} models from LMSYS Chatbot Arena")
    except Exception as e:
        logger.warning(f"Failed to fetch LMSYS leaderboard: {e}")

    if not all_benchmarks:
        logger.warning("No benchmark data fetched, using cache")
        return

    _cache = all_benchmarks
    _cache_time = datetime.now(timezone.utc)

    async with async_session() as db:
        from sqlalchemy import delete
        await db.execute(delete(ModelBenchmark))

        now = datetime.now(timezone.utc)
        for b in all_benchmarks:
            db.add(ModelBenchmark(
                source=b["source"],
                model_name=b["model_name"],
                overall_score=b["overall_score"],
                domain_scores=b["domain_scores"],
                fetched_at=now,
            ))
        await db.commit()

    logger.info(f"Stored {len(all_benchmarks)} benchmark entries")
