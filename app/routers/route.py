import time
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.ai_model import AIModel
from app.models.request_log import RequestLog
from app.models.user import User
from app.schemas.route import RouteRequest, RoutingStrategy
from app.services.model_caller import call_model, CircuitOpenError
from app.services.router import select_model
from app.services.adaptive import adaptive_select, thompson_select

logger = logging.getLogger(__name__)

router = APIRouter(tags=["路由"])


@router.post("/route")
async def route_request(
    body: RouteRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(AIModel).where(AIModel.user_id == user.id))
    models = list(result.scalars().all())

    if not models:
        raise HTTPException(status_code=404, detail="没有可用的模型配置")

    messages = [m.model_dump() for m in body.messages]
    strategy = body.strategy
    benchmark_score = None

    realtime_eval = body.enable_realtime_eval
    if realtime_eval is None:
        realtime_eval = settings.ENABLE_REALTIME_EVAL

    if strategy == RoutingStrategy.ADAPTIVE and realtime_eval:
        chosen, eval_results, eval_duration = await adaptive_select(
            models, messages, body.preferred_labels, user.id
        )
        if not chosen:
            raise HTTPException(status_code=404, detail="自适应路由未找到可用模型")

        logger.debug(
            "adaptive eval: selected=%s duration=%.1fms candidates=%s",
            chosen.name,
            eval_duration * 1000,
            [(r.model.name, r.combined_score, r.cost_adjusted_score) for r in eval_results],
        )
        benchmark_score = max(r.combined_score for r in eval_results) if eval_results else None

    elif strategy == RoutingStrategy.THOMPSON:
        chosen, ts_score = thompson_select(models, body.preferred_labels, user.id)
        if not chosen:
            raise HTTPException(status_code=404, detail="Thompson采样未找到可用模型")
        logger.debug("thompson: selected=%s score=%.4f", chosen.name, ts_score)
        benchmark_score = ts_score

    elif strategy == RoutingStrategy.STATIC:
        chosen = min(models, key=lambda m: m.cost_per_1k_tokens)
        logger.debug("static: selected=%s", chosen.name)

    else:
        chosen, benchmark_score = select_model(models, body.preferred_labels)
        if not chosen:
            raise HTTPException(status_code=404, detail="没有可用的模型配置")
        logger.debug("leaderboard: selected=%s score=%s", chosen.name, benchmark_score)

    start = time.perf_counter()
    try:
        response_data = await call_model(
            api_url=chosen.api_url,
            api_key=chosen.api_key,
            model_name=chosen.name,
            messages=messages,
            model_id=chosen.id,
        )
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"模型调用失败: {str(e)}")

    duration_ms = (time.perf_counter() - start) * 1000

    total_tokens = 0
    usage = response_data.get("usage")
    if usage:
        total_tokens = usage.get("total_tokens", 0)
    cost = (total_tokens / 1000) * chosen.cost_per_1k_tokens

    log = RequestLog(
        user_id=user.id,
        model_id=chosen.id,
        model_name=chosen.name,
        duration_ms=duration_ms,
        total_tokens=total_tokens,
        cost=cost,
        benchmark_score=benchmark_score,
    )
    db.add(log)
    await db.commit()

    return JSONResponse(content=response_data)
