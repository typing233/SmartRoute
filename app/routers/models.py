from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.ai_model import AIModel
from app.models.user import User
from app.schemas.ai_model import AIModelCreate, AIModelOut

router = APIRouter(prefix="/models", tags=["模型管理"])


@router.get("/", response_model=list[AIModelOut])
async def list_models(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(AIModel).where(AIModel.user_id == user.id))
    return result.scalars().all()


@router.post("/", response_model=AIModelOut, status_code=status.HTTP_201_CREATED)
async def create_model(
    body: AIModelCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    model = AIModel(
        user_id=user.id,
        name=body.name,
        api_url=body.api_url,
        api_key=body.api_key,
        cost_per_1k_tokens=body.cost_per_1k_tokens,
        labels=body.labels,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AIModel).where(AIModel.id == model_id, AIModel.user_id == user.id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    await db.delete(model)
    await db.commit()
