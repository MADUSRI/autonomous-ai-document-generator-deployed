# Autonomous AI Agent for Business Document Generation

This project implements a simple FastAPI service that accepts a natural language request, creates a task plan, executes the planning steps, and generates a polished Microsoft Word document (.docx) as output.

## Features
- POST /agent accepts JSON: {"request": "..."}
- Autonomous multi-step planning
- Deterministic document generation with Python-docx
- Optional local LLM fallback when Ollama is running on localhost:11434
- Validation and clear API responses

## Run locally
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Start the API:
   - `python start_server.py`
3. Send a test request:
   - `curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" -d '{"request":"Create a concise project proposal for a new AI scheduling assistant for small clinics."}'`

## LLM provider configuration
- Local development (Ollama + Llama 3.2):
  - `LLM_PROVIDER=ollama`
  - `OLLAMA_URL=http://localhost:11434`
  - `OLLAMA_MODEL=llama3.2:latest`
- Cloud deployment (OpenRouter API):
  - `LLM_PROVIDER=openrouter`
  - `OPENROUTER_API_KEY=<your_openrouter_api_key>`
  - `OPENROUTER_MODEL=openai/gpt-4o`
  - `OPENROUTER_API_URL=https://openrouter.ai/api/v1`

Example local startup:
- `LLM_PROVIDER=ollama OLLAMA_URL=http://localhost:11434 python start_server.py`

Example cloud startup:
- `LLM_PROVIDER=groq GROQ_API_KEY=your_key GROQ_MODEL=llama uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`

## Deploy for anyone with the link
- The app now exposes a health check at `/health` and binds to `0.0.0.0` by default.
- On a host such as Render, Railway, or Fly.io, set `PORT` to the platform-assigned port.
- Example startup command:
  - `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`

## Engineering improvement implemented
- Multi-step planning: the agent breaks the request into a structured execution plan before generating the document. This improves autonomy and makes the workflow more robust for ambiguous or complex requests.

## Verification
The project has been verified with:
- `pytest -q`
