import json
import re
import time
from typing import Any, Dict, List

import httpx

from .config import get_settings

settings = get_settings()
LLM_PROVIDER = settings.LLM_PROVIDER.strip().lower()
OLLAMA_URL = settings.OLLAMA_URL.rstrip("/")
OLLAMA_MODEL = settings.OLLAMA_MODEL.strip()
GROQ_API_URL = settings.GROQ_API_URL.rstrip("/")
GROQ_API_KEY = settings.GROQ_API_KEY.strip()
GROQ_MODEL = settings.GROQ_MODEL.strip()
GROQ_MAX_OUTPUT_TOKENS = settings.GROQ_MAX_OUTPUT_TOKENS
OPENROUTER_API_URL = settings.OPENROUTER_API_URL.rstrip("/")
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY.strip()
OPENROUTER_MODEL = settings.OPENROUTER_MODEL.strip()


def build_prompt_template(doc_type: str, request: str, subject: str) -> str:
    return (
        "You are an AI project planner. Your job is only to plan work and decompose it into tasks. "
        "Do not generate document content. Determine the most suitable document type, identify assumptions, "
        "and decompose the request into executable tasks. Return strict JSON with keys document_type, assumptions, tasks, summary. "
        "Each task must be an object with id, title, and description. "
        "Create a professional, document-type-specific task list that reflects the conventions of the target document. "
        "Prefer expert-level section headings and workstreams that are appropriate for the document type rather than generic tasks. "
        "For example, a Technical Design should include sections such as Executive Summary, Functional Requirements, Non-functional Requirements, High-Level Architecture, Component Design, Database Design, API Design, Security, Deployment, Risks, and Recommendations. "
        "A Business Proposal should include Executive Summary, Objectives, Scope, Deliverables, Timeline, Budget, Risks, and Recommendations. "
        "Meeting Minutes should include Attendees, Agenda, Discussion Summary, Decisions, Action Items, Owners, and Next Meeting. "
        "An SOP should include Purpose, Scope, Responsibilities, Procedure, Exceptions, and References. "
        "Use the document type and user request to choose the most relevant structure, and keep tasks actionable and professional. "
        f"Document type hint: {doc_type}\n"
        f"User request: {request}\n"
        f"Subject: {subject}\n"
        "Return only the JSON object."
    )


def _extract_json_payload(content: str) -> Dict[str, Any]:
    text = content.strip()
    if not text:
        return {}

    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _parse_ollama_response(response_json: Any) -> str:
    if isinstance(response_json, dict):
        response_text = response_json.get("response")
        if isinstance(response_text, str):
            return response_text.strip()
    return ""


def _parse_groq_response(response_json: Any) -> str:
    if isinstance(response_json, dict):
        if "output" in response_json:
            output = response_json["output"]
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict):
                    for key in ("content", "text", "message", "output"):
                        value = first.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                if isinstance(first, str) and first.strip():
                    return first.strip()
        for key in ("response", "text", "output", "content"):
            value = response_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _parse_openrouter_response(response_json: Any) -> str:
    if isinstance(response_json, dict):
        choices = response_json.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                for key in ("content", "text", "message"):
                    value = first_choice.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        for key in ("response", "text", "content"):
            value = response_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _ollama_request(prompt: str, model: str, timeout: float) -> str:
    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama request failed: {response.status_code} {response.text}")
    return _parse_ollama_response(response.json())


def _groq_request(prompt: str, model: str, timeout: float) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be set when LLM_PROVIDER is groq.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        f"{GROQ_API_URL}/models/{model}/infer",
        headers=headers,
        json={
            "input": prompt,
            "max_output_tokens": GROQ_MAX_OUTPUT_TOKENS,
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Groq request failed: {response.status_code} {response.text}")
    return _parse_groq_response(response.json())


def _is_provider_available(provider: str) -> bool:
    if provider == "ollama":
        return bool(OLLAMA_URL)
    if provider == "groq":
        return bool(GROQ_API_KEY and GROQ_API_URL)
    if provider == "openrouter":
        return bool(OPENROUTER_API_KEY and OPENROUTER_API_URL)
    return False


def _call_llm_provider(provider: str, prompt: str, model: str | None, timeout: float) -> str:
    if provider == "ollama":
        selected_model = model or OLLAMA_MODEL
        return _ollama_request(prompt, selected_model, timeout)
    if provider == "groq":
        selected_model = model or GROQ_MODEL
        return _groq_request(prompt, selected_model, timeout)
    if provider == "openrouter":
        selected_model = model or OPENROUTER_MODEL
        return _openrouter_request(prompt, selected_model, timeout)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")


def send_llm_request(prompt: str, model: str | None = None, timeout: float = 600.0) -> str:
    primary_provider = LLM_PROVIDER
    provider_order = [primary_provider] + [provider for provider in ("openrouter", "groq", "ollama") if provider != primary_provider]

    last_error: Exception | None = None
    for index, provider in enumerate(provider_order):
        if not _is_provider_available(provider):
            continue

        try:
            provider_model = model if provider == primary_provider else None
            return _call_llm_provider(provider, prompt, provider_model, timeout)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RequestError) as exc:
            last_error = exc
            if provider == primary_provider and any(_is_provider_available(p) for p in provider_order[index + 1 :]):
                print(f"LLM provider {provider} timed out or network error; falling back to the next configured provider.")
                continue
            raise RuntimeError(f"LLM request failed for provider {provider}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM request failed for provider {provider}: {exc}") from exc

    if last_error:
        raise RuntimeError(f"No configured LLM provider succeeded. Last error: {last_error}") from last_error
    raise RuntimeError(f"No configured LLM provider available for {LLM_PROVIDER}.")


def _build_task_execution_prompt(
    request: str,
    doc_type: str,
    current_task: Dict[str, Any],
    previous_sections: List[Dict[str, Any]],
    assumptions: List[str],
) -> str:
    title = str(current_task.get("title") or current_task.get("name") or "Task").strip()
    description = str(current_task.get("description") or current_task.get("details") or "").strip()

    previous_context = ""
    if previous_sections:
        previous_context = "Previously completed sections:\n" + json.dumps(previous_sections, indent=2)

    assumptions_context = ""
    if assumptions:
        assumptions_context = "Assumptions:\n" + "\n".join(str(item).strip() for item in assumptions if str(item).strip()) + "\n"

    return (
        "You are an experienced business consultant and technical writer executing one step in a larger autonomous workflow. "
        "Your job is to generate content ONLY for the current task section of a business document. "
        "Use the original user request, the document type, all previously completed sections, and all assumptions to create detailed, professional, domain-specific content. "
        "Do not generate the entire document and do not repeat earlier sections. Avoid generic placeholders and vague language. "
        "Make reasonable assumptions when information is missing, but keep the content practical and relevant to the user's request. "
        "When appropriate, include bullet points, numbered lists, tables, implementation steps, decision criteria, or risk considerations. "
        "Return only the content for the requested section, without headings or prefatory commentary.\n"
        f"Document type: {doc_type}\n"
        f"Original request: {request}\n"
        f"Current task title: {title}\n"
        f"Current task description: {description}\n"
        f"{assumptions_context}"
        f"{previous_context}\n"
        "Generate only the content needed for this task."
    )


def execute_task(
    request: str,
    doc_type: str,
    current_task: Dict[str, Any],
    previous_sections: List[Dict[str, Any]],
    assumptions: List[str] | None = None,
) -> str:
    prompt = _build_task_execution_prompt(request, doc_type, current_task, previous_sections, assumptions or [])
    start = time.time()
    try:
        payload = send_llm_request(prompt, timeout=600.0)
        if payload:
            return payload
        print("Time:", time.time() - start)
    except Exception as e:
        print("Time:", time.time() - start)
        import traceback
        traceback.print_exc()
        print(f"LLM Error: {e}")

    title = str(current_task.get("title") or current_task.get("name") or "Task").strip()
    description = str(current_task.get("description") or current_task.get("details") or "").strip()
    if description:
        return f"Content for {title}: {description}"
    return f"Content for {title}: the execution engine should expand this section with task-specific detail."


def try_local_llm(request: str, doc_type: str) -> Dict[str, Any]:
    start = time.time()
    try:
        from .planner import infer_subject

        subject = infer_subject(request)
        response_text = send_llm_request(build_prompt_template(doc_type, request, subject), timeout=600.0)
        if response_text:
            print("Time:", time.time() - start)
            payload = _extract_json_payload(response_text)
            if isinstance(payload, dict):
                assumptions = payload.get("assumptions")
                tasks = payload.get("tasks")
                if isinstance(assumptions, list):
                    normalized_assumptions = [str(item).strip() for item in assumptions if str(item).strip()]
                elif isinstance(assumptions, str) and assumptions.strip():
                    normalized_assumptions = [assumptions.strip()]
                else:
                    normalized_assumptions = []

                if isinstance(tasks, list):
                    normalized_tasks = tasks
                else:
                    normalized_tasks = []

                return {
                    "document_type": str(payload.get("document_type") or doc_type or "Business Proposal").strip(),
                    "assumptions": normalized_assumptions,
                    "tasks": normalized_tasks,
                    "summary": str(payload.get("summary") or f"Planned the work for a {doc_type.lower()}.").strip(),
                }
    except Exception as e:
        print("Time:", time.time() - start)
        import traceback
        traceback.print_exc()
        print(f"LLM Error: {e}")
    return {}
