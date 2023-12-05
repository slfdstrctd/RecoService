from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from saved_models.models import userknn


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name == "some_model":
        reco = list(range(k_recs))
    elif model_name == "userknn":
        reco = userknn.recommend(user_id=user_id, N_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
