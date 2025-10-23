from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.service.answer import answer_question

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask(request: QuestionRequest):
    """Pose une question en langage naturel sur les évènements publics prévus"""
    return answer_question(request.question)