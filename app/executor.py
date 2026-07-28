from typing import Any, Dict, List, Sequence

from .llm import execute_task
import logging


def _normalize_tasks(plan: Sequence[Any] | None) -> List[Dict[str, Any]]:
    if not plan:
        return []

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or f"Task {index}").strip()
            description = str(item.get("description") or item.get("details") or "").strip()
            if title:
                normalized.append({"title": title, "description": description})
        elif isinstance(item, str) and item.strip():
            normalized.append({"title": item.strip(), "description": ""})
    return normalized


def execute_document_generation(request: str, doc_type: str, plan: Sequence[Any], assumptions: List[str]) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "request": request,
        "document_type": doc_type,
        "assumptions": assumptions,
        "sections": [],
    }

    previous_sections: List[Dict[str, Any]] = []
    for task in _normalize_tasks(plan):
        generated_content = execute_task(request, doc_type, task, previous_sections, assumptions)
        print(f"Generated content for task '{task['title']}': {generated_content}")
        logging.info(
            "Generated section '%s'",
            task["title"]
        )
        section = {
            "title": task["title"],
            "content": generated_content,
        }
        context["sections"].append(section)
        previous_sections.append(section)

    return context
