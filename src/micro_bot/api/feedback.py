import json

from fastapi import APIRouter, Request

from micro_bot.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    data_dir = request.app.state.micro_bot_state.settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "feedback.jsonl").open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(payload.model_dump(), ensure_ascii=False) + "\n")
    return FeedbackResponse()
