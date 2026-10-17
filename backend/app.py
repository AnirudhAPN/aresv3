"""
ARES FastAPI application.

Uses a background-task + polling pattern (POST to start, GET to poll status,
GET to fetch the full report) rather than a single blocking endpoint, since
five sequential (and potentially retried) agent calls plus external API
calls would risk client timeouts.
"""
from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from config import settings
from models.schemas import (
    AssistantChatRequest,
    AssistantChatResponse,
    ChatRequest,
    ChatResponse,
    ResearchRequest,
    ResearchStartResponse,
    ResearchStatusResponse,
)
from orchestrator import run_pipeline
from services.memory import memory_service
from services import assistant as assistant_service
from services import export as export_service
from services import chat as chat_service

app = FastAPI(
    title="ARES — Adaptive Research Ecosystem Suite",
    description="AI-powered multi-agent research assistant grounded in real literature.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/research", response_model=ResearchStartResponse)
def start_research(req: ResearchRequest, background_tasks: BackgroundTasks) -> ResearchStartResponse:
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")

    topic = req.topic.strip()
    similar_sessions = memory_service.find_similar_sessions(topic)
    session_id = memory_service.create_session(topic)
    background_tasks.add_task(run_pipeline, session_id, topic, "adaptive")
    return ResearchStartResponse(session_id=session_id, status="started", similar_sessions=similar_sessions)


@app.get("/api/research/{session_id}/status", response_model=ResearchStatusResponse)
def get_status(session_id: str) -> ResearchStatusResponse:
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return ResearchStatusResponse(
        session_id=session_id,
        status=session["status"],
        current_agent=session.get("current_agent"),
        completed_agents=session.get("completed_agents", []),
        acs_scores=session.get("acs_scores", {}),
        acs_components=session.get("acs_components", {}),
        coherence=session.get("coherence"),
        flags=session.get("flags", []),
    )


@app.get("/api/research/{session_id}")
def get_report(session_id: str) -> dict:
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.get("/api/history")
def get_history() -> list[dict]:
    return memory_service.list_history()


@app.get("/api/history/{session_id}")
def get_history_item(session_id: str) -> dict:
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.post("/api/research/{session_id}/chat", response_model=ChatResponse)
def post_chat_message(session_id: str, req: ChatRequest) -> ChatResponse:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        result = chat_service.handle_chat_message(session_id, req.message.strip())
    except ValueError:
        raise HTTPException(status_code=404, detail="session not found")
    return ChatResponse(**result)


@app.get("/api/research/{session_id}/chat")
def get_chat_history(session_id: str) -> list[dict]:
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return memory_service.get_chat_history(session_id)


@app.post("/api/assistant/chat", response_model=AssistantChatResponse)
def post_assistant_message(req: AssistantChatRequest) -> AssistantChatResponse:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    result = assistant_service.chat(req.message.strip())
    return AssistantChatResponse(**result)


@app.get("/api/assistant/chat")
def get_assistant_history() -> list[dict]:
    return memory_service.get_assistant_history()


@app.delete("/api/assistant/chat")
def clear_assistant_history() -> dict:
    memory_service.clear_assistant_history()
    return {"status": "cleared"}


@app.get("/api/research/{session_id}/export/markdown")
def export_markdown(session_id: str):
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return PlainTextResponse(export_service.to_markdown(session), media_type="text/markdown")


@app.get("/api/research/{session_id}/export/text")
def export_text(session_id: str):
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return PlainTextResponse(export_service.to_text(session), media_type="text/plain")


@app.get("/api/research/{session_id}/export/pdf")
def export_pdf(session_id: str):
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        pdf_bytes = export_service.render_pdf_bytes(session)
    except (ImportError, OSError) as exc:
        # WeasyPrint requires native GTK/Pango libraries; when they're
        # missing it raises OSError from cffi's dlopen, not ImportError —
        # both mean "PDF export unavailable on this host," not a 500.
        logging.error("PDF export failed — missing native dependency: %s", exc)
        raise HTTPException(status_code=501, detail="PDF export is not available on this server right now.")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.pdf"'},
    )


@app.get("/api/research/{session_id}/export/docx")
def export_docx(session_id: str):
    session = memory_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        docx_bytes = export_service.render_docx_bytes(session)
    except ImportError as exc:
        logging.error("Word export failed — missing dependency: %s", exc)
        raise HTTPException(status_code=501, detail="Word export is not available on this server right now.")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.docx"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
