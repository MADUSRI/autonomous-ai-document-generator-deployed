import json
import re
import time
from typing import Any, Dict, List

import httpx
from .llm import send_llm_request


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


def reflect_on_output(execution_context: Any, assumptions: List[str] | None = None, summary: str | None = None) -> Dict[str, Any]:
    if isinstance(execution_context, dict) and ("request" in execution_context or "sections" in execution_context):
        context = execution_context
    else:
        context = {
            "request": "",
            "document_type": "Document",
            "assumptions": assumptions or [],
            "sections": [{"title": str(item), "content": ""} for item in (execution_context or [])],
        }

    request = str(context.get("request") or "").strip()
    doc_type = str(context.get("document_type") or "Document").strip()
    assumptions = context.get("assumptions") or []
    sections = context.get("sections") or []

    prompt = (
        "You are a senior business analyst and technical editor reviewing the complete execution context for a document generation workflow. "
        "Identify missing sections, detect inconsistencies, remove repeated information, improve clarity, improve professionalism, and ensure every section directly supports the user's request. "
        "Return strict JSON with keys request, document_type, assumptions, and sections. "
        "The sections array must contain improved section objects with title and content. "
        "Use the full execution context and preserve the document's purpose. Return only the JSON object.\n"
        f"Execution context: {json.dumps(context, indent=2)}"
    )
    start = time.time()
    try:
        # response = httpx.post(
        #     "http://localhost:11434/api/generate",
        #     json={
        #         "model": "llama3.2:latest",
        #         "prompt": prompt,
        #         "stream": False,
        #     },
        #     timeout=600.0,
        # )
        response_text = send_llm_request(
            prompt,
            timeout=600.0
        )

        payload = _extract_json_payload(response_text)
        # if response.status_code == 200:
        #     payload = _extract_json_payload(response.json().get("response", ""))
        #     if isinstance(payload, dict):
        #         improved_sections = payload.get("sections")
        #         if isinstance(improved_sections, list):
        #             normalized_sections = []
        #             for section in improved_sections:
        #                 if isinstance(section, dict):
        #                     title = str(section.get("title") or "Section").strip()
        #                     content = str(section.get("content") or "").strip()
        #                     if title:
        #                         normalized_sections.append({"title": title, "content": content})
        #             if normalized_sections:
        #                 return {
        #                     "plan": [section["title"] for section in normalized_sections],
        #                     "assumptions": [str(item).strip() for item in payload.get("assumptions", assumptions) if str(item).strip()],
        #                     "summary": summary or f"Refined the execution context for a {doc_type.lower()}.",
        #                     "execution_context": {
        #                         "request": str(payload.get("request") or request),
        #                         "document_type": str(payload.get("document_type") or doc_type),
        #                         "assumptions": [str(item).strip() for item in payload.get("assumptions", assumptions) if str(item).strip()],
        #                         "sections": normalized_sections,
        #                     },
        #                 }
        print("Time:", time.time() - start)
    except Exception as e:
        print("Time:", time.time() - start)
        import traceback
        traceback.print_exc()
        print(f"LLM Error: {e}")
        pass

    normalized_sections = []
    for section in sections:
        if isinstance(section, dict):
            title = str(section.get("title") or "Section").strip()
            content = str(section.get("content") or "").strip()
            if title:
                normalized_sections.append({"title": title, "content": content})

    return {
        "plan": [section["title"] for section in normalized_sections],
        "assumptions": [str(item).strip() for item in assumptions if str(item).strip()],
        "summary": summary or f"Refined the execution context for a {doc_type.lower()}.",
        "execution_context": {
            "request": request,
            "document_type": doc_type,
            "assumptions": [str(item).strip() for item in assumptions if str(item).strip()],
            "sections": normalized_sections,
        },
    }
