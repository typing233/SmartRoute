import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers import auth, models, route, health, report
from app.services.scheduler import start_scheduler, stop_scheduler
from app.services.leaderboard import fetch_and_store_benchmarks

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    asyncio.create_task(fetch_and_store_benchmarks())
    yield
    stop_scheduler()


app = FastAPI(
    title="SmartRoute 睿路由",
    description="AI 模型智能路由服务 —— 根据能力标签、排行榜得分和成本自动选择最优模型",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(models.router)
app.include_router(route.router)
app.include_router(report.router)
