"""
Integration test: compare static, leaderboard, and adaptive routing strategies.
Simulates 100 requests with mock LLM endpoints to measure total cost and quality scores.

Usage:
    python tests/test_routing_strategies.py
"""

import asyncio
import random
import time
import json
from dataclasses import dataclass, field
from unittest.mock import patch, AsyncMock
from collections import defaultdict

import sys
sys.path.insert(0, ".")

from app.services.router import select_model
from app.services.adaptive import (
    adaptive_select, thompson_select, _history, ModelHistory,
)
from app.services.leaderboard import _cache


# --- Lightweight mock model (avoids SQLAlchemy instrumentation) ---

class MockModel:
    def __init__(self, id, user_id, name, api_url, api_key, cost_per_1k_tokens, labels):
        self.id = id
        self.user_id = user_id
        self.name = name
        self.api_url = api_url
        self.api_key = api_key
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.labels = labels


# --- Mock model definitions ---

MOCK_MODELS = [
    {"name": "gpt-4o", "cost": 0.03, "quality": 0.92, "labels": ["reasoning", "code", "creative"]},
    {"name": "gpt-4o-mini", "cost": 0.005, "quality": 0.78, "labels": ["math", "knowledge"]},
    {"name": "claude-sonnet-4-6", "cost": 0.015, "quality": 0.88, "labels": ["reasoning", "creative", "code"]},
    {"name": "claude-haiku-4-5", "cost": 0.003, "quality": 0.72, "labels": ["knowledge", "math"]},
    {"name": "deepseek-v3", "cost": 0.002, "quality": 0.75, "labels": ["code", "math", "reasoning"]},
]

SCENARIOS = [
    {"prompt": "Explain quantum computing", "labels": ["knowledge", "science"]},
    {"prompt": "Write a Python sorting algorithm", "labels": ["code"]},
    {"prompt": "Solve this math problem: integral of x^2", "labels": ["math"]},
    {"prompt": "Write a creative story about AI", "labels": ["creative"]},
    {"prompt": "Debug this code snippet", "labels": ["code", "reasoning"]},
    {"prompt": "Summarize this article", "labels": []},
    {"prompt": "Translate this to French", "labels": []},
    {"prompt": "What is the capital of France?", "labels": ["knowledge"]},
    {"prompt": "Prove the Pythagorean theorem", "labels": ["math", "reasoning"]},
    {"prompt": "Write a haiku about programming", "labels": ["creative"]},
]


@dataclass
class SimulatedResult:
    strategy: str
    model_name: str
    quality_score: float
    cost: float
    latency_ms: float


def _create_mock_model(spec: dict, model_id: int, user_id: int = 1) -> MockModel:
    return MockModel(
        id=model_id,
        user_id=user_id,
        name=spec["name"],
        api_url=f"http://mock/{spec['name']}",
        api_key="mock-key",
        cost_per_1k_tokens=spec["cost"],
        labels=spec["labels"],
    )


def _simulate_quality(model_spec: dict, scenario: dict) -> float:
    base = model_spec["quality"]
    label_bonus = 0.0
    if scenario["labels"]:
        matching = set(scenario["labels"]) & set(model_spec["labels"])
        label_bonus = 0.05 * len(matching)
    noise = random.gauss(0, 0.03)
    return max(0.0, min(1.0, base + label_bonus + noise))


def _simulate_token_count() -> int:
    return random.randint(100, 500)


def _setup_mock_benchmarks():
    global _cache
    benchmarks = []
    for spec in MOCK_MODELS:
        benchmarks.append({
            "model_name": spec["name"],
            "source": "mock",
            "overall_score": spec["quality"] * 100,
            "domain_scores": {
                "math": spec["quality"] * 100 + random.uniform(-5, 5),
                "reasoning": spec["quality"] * 100 + random.uniform(-5, 5),
                "knowledge": spec["quality"] * 100 + random.uniform(-5, 5),
                "code": spec["quality"] * 100 + random.uniform(-5, 5),
                "creative": spec["quality"] * 100 + random.uniform(-5, 5),
                "science": spec["quality"] * 100 + random.uniform(-5, 5),
            },
        })
    _cache.clear()
    _cache.extend(benchmarks)


async def run_static_routing(models, scenarios, num_requests):
    results = []
    for i in range(num_requests):
        scenario = scenarios[i % len(scenarios)]
        chosen = min(models, key=lambda m: m.cost_per_1k_tokens)
        spec = next(s for s in MOCK_MODELS if s["name"] == chosen.name)
        quality = _simulate_quality(spec, scenario)
        tokens = _simulate_token_count()
        cost = (tokens / 1000) * chosen.cost_per_1k_tokens
        results.append(SimulatedResult(
            strategy="static",
            model_name=chosen.name,
            quality_score=quality,
            cost=cost,
            latency_ms=random.uniform(200, 800),
        ))
    return results


async def run_leaderboard_routing(models, scenarios, num_requests):
    results = []
    for i in range(num_requests):
        scenario = scenarios[i % len(scenarios)]
        chosen, _ = select_model(models, scenario["labels"])
        if not chosen:
            chosen = models[0]
        spec = next(s for s in MOCK_MODELS if s["name"] == chosen.name)
        quality = _simulate_quality(spec, scenario)
        tokens = _simulate_token_count()
        cost = (tokens / 1000) * chosen.cost_per_1k_tokens
        results.append(SimulatedResult(
            strategy="leaderboard",
            model_name=chosen.name,
            quality_score=quality,
            cost=cost,
            latency_ms=random.uniform(200, 800),
        ))
    return results


async def run_adaptive_routing(models, scenarios, num_requests):
    results = []
    user_id = 1

    async def mock_generate_snippet(api_url, api_key, model_name, messages):
        spec = next((s for s in MOCK_MODELS if s["name"] == model_name), None)
        if spec:
            return f"Mock snippet from {model_name}", {"total_tokens": 30}
        return "", None

    async def mock_evaluate_snippet(snippet, user_prompt):
        if not snippet:
            return 0.0, 0.0
        model_name = snippet.replace("Mock snippet from ", "")
        spec = next((s for s in MOCK_MODELS if s["name"] == model_name), None)
        if spec:
            relevance = spec["quality"] + random.gauss(0, 0.05)
            fluency = spec["quality"] + random.gauss(0, 0.03)
            return max(0, min(1, relevance)), max(0, min(1, fluency))
        return 0.5, 0.5

    with patch("app.services.adaptive.generate_snippet", side_effect=mock_generate_snippet):
        with patch("app.services.adaptive.evaluate_snippet", side_effect=mock_evaluate_snippet):
            for i in range(num_requests):
                scenario = scenarios[i % len(scenarios)]
                messages = [{"role": "user", "content": scenario["prompt"]}]

                chosen, eval_results, eval_duration = await adaptive_select(
                    models, messages, scenario["labels"], user_id
                )
                if not chosen:
                    chosen = models[0]

                spec = next(s for s in MOCK_MODELS if s["name"] == chosen.name)
                quality = _simulate_quality(spec, scenario)
                tokens = _simulate_token_count()
                cost = (tokens / 1000) * chosen.cost_per_1k_tokens
                eval_token_cost = (30 * len(eval_results) / 1000) * 0.005

                results.append(SimulatedResult(
                    strategy="adaptive",
                    model_name=chosen.name,
                    quality_score=quality,
                    cost=cost + eval_token_cost,
                    latency_ms=random.uniform(200, 800) + eval_duration * 1000,
                ))
    return results


async def run_thompson_routing(models, scenarios, num_requests):
    results = []
    user_id = 2  # separate history from adaptive

    for i in range(num_requests):
        scenario = scenarios[i % len(scenarios)]
        chosen, _ = thompson_select(models, scenario["labels"], user_id)
        if not chosen:
            chosen = models[0]
        spec = next(s for s in MOCK_MODELS if s["name"] == chosen.name)
        quality = _simulate_quality(spec, scenario)
        tokens = _simulate_token_count()
        cost = (tokens / 1000) * chosen.cost_per_1k_tokens

        # Update thompson history with observed quality
        from app.services.adaptive import get_history, _label_key
        label_key = _label_key(scenario["labels"])
        history = get_history(user_id, chosen.name, label_key)
        history.update(quality)

        results.append(SimulatedResult(
            strategy="thompson",
            model_name=chosen.name,
            quality_score=quality,
            cost=cost,
            latency_ms=random.uniform(200, 800),
        ))
    return results


def print_report(all_results: dict[str, list[SimulatedResult]]):
    print("\n" + "=" * 80)
    print("ROUTING STRATEGY COMPARISON REPORT")
    print(f"Requests per strategy: 100")
    print("=" * 80)

    headers = f"{'Strategy':<15} {'Avg Quality':<13} {'Total Cost':<12} {'Avg Cost':<11} {'Avg Latency':<13} {'Model Distribution'}"
    print(f"\n{headers}")
    print("-" * 80)

    for strategy_name, results in all_results.items():
        avg_quality = sum(r.quality_score for r in results) / len(results)
        total_cost = sum(r.cost for r in results)
        avg_cost = total_cost / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        model_counts = defaultdict(int)
        for r in results:
            model_counts[r.model_name] += 1
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(model_counts.items(), key=lambda x: -x[1]))

        print(f"{strategy_name:<15} {avg_quality:<13.4f} ${total_cost:<11.4f} ${avg_cost:<10.4f} {avg_latency:<13.1f} {dist}")

    print("\n" + "-" * 80)
    print("ANALYSIS:")

    strategies = list(all_results.keys())
    qualities = {s: sum(r.quality_score for r in rs) / len(rs) for s, rs in all_results.items()}
    costs = {s: sum(r.cost for r in rs) for s, rs in all_results.items()}

    best_quality = max(qualities, key=qualities.get)
    lowest_cost = min(costs, key=costs.get)

    print(f"  Best quality:  {best_quality} (avg score: {qualities[best_quality]:.4f})")
    print(f"  Lowest cost:   {lowest_cost} (total: ${costs[lowest_cost]:.4f})")

    if "adaptive" in all_results and "static" in all_results:
        q_improvement = (qualities["adaptive"] - qualities["static"]) / qualities["static"] * 100
        cost_diff = (costs["adaptive"] - costs["static"]) / costs["static"] * 100
        print(f"  Adaptive vs Static: quality {q_improvement:+.1f}%, cost {cost_diff:+.1f}%")

    if "adaptive" in all_results and "leaderboard" in all_results:
        q_improvement = (qualities["adaptive"] - qualities["leaderboard"]) / qualities["leaderboard"] * 100
        cost_diff = (costs["adaptive"] - costs["leaderboard"]) / costs["leaderboard"] * 100
        print(f"  Adaptive vs Leaderboard: quality {q_improvement:+.1f}%, cost {cost_diff:+.1f}%")

    print("=" * 80)


async def main():
    random.seed(42)
    print("Setting up mock environment...")

    models = [_create_mock_model(spec, idx + 1) for idx, spec in enumerate(MOCK_MODELS)]
    _setup_mock_benchmarks()

    # Clear adaptive history
    _history.clear()

    num_requests = 100
    print(f"Running {num_requests} simulated requests per strategy...\n")

    print("[1/4] Running static routing (cheapest model)...")
    static_results = await run_static_routing(models, SCENARIOS, num_requests)

    print("[2/4] Running leaderboard routing (benchmark-based)...")
    leaderboard_results = await run_leaderboard_routing(models, SCENARIOS, num_requests)

    print("[3/4] Running adaptive routing (real-time eval)...")
    adaptive_results = await run_adaptive_routing(models, SCENARIOS, num_requests)

    print("[4/4] Running Thompson Sampling routing...")
    thompson_results = await run_thompson_routing(models, SCENARIOS, num_requests)

    all_results = {
        "static": static_results,
        "leaderboard": leaderboard_results,
        "adaptive": adaptive_results,
        "thompson": thompson_results,
    }

    print_report(all_results)

    # Output JSON summary for programmatic use
    summary = {}
    for strategy_name, results in all_results.items():
        summary[strategy_name] = {
            "avg_quality": sum(r.quality_score for r in results) / len(results),
            "total_cost": sum(r.cost for r in results),
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        }

    with open("tests/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to tests/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
