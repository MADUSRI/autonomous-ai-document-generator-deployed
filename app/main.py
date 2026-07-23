import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .agent import run_agent

app = FastAPI(title="Autonomous Agent API")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated_docs"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


class AgentRequest(BaseModel):
    request: str = Field(..., min_length=3, max_length=4000)


class AgentResponse(BaseModel):
    message: str
    plan: List[str]
    assumptions: List[str]
    document_name: str
    document_path: str
    summary: str
    planning_mode: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=filename)


@app.post("/agent", response_model=AgentResponse)
def agent(payload: AgentRequest) -> Dict[str, Any]:
    request_text = payload.request.strip()
    if not request_text:
        raise HTTPException(status_code=400, detail="Request cannot be empty")

    return run_agent(request_text)

