from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers import auth, models, route, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="SmartRoute 睿路由",
    description="AI 模型智能路由服务 —— 根据能力标签和成本自动选择最优模型",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(models.router)
app.include_router(route.router)
