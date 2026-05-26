from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.request_log import RequestLog
from app.models.user import User

router = APIRouter(tags=["报表"])


@router.get("/report")
async def get_report(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(
            func.count(RequestLog.id).label("request_count"),
            func.coalesce(func.sum(RequestLog.cost), 0).label("total_cost"),
            func.avg(RequestLog.benchmark_score).label("average_score"),
        ).where(RequestLog.user_id == user.id)
    )
    row = result.one()

    return {
        "request_count": row.request_count,
        "total_cost": round(float(row.total_cost), 6),
        "average_score": round(float(row.average_score), 2) if row.average_score else None,
    }
