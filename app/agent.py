from typing import Any, Dict, List

from .document_generator import build_document_from_context
from .executor import execute_document_generation
from .planner import build_plan, detect_document_type
from .reflector import reflect_on_output


def _format_tasks(tasks: List[Dict[str, Any]]) -> List[str]:
    formatted: List[str] = []
    for task in tasks:
        if isinstance(task, dict):
            title = str(task.get("title") or task.get("name") or "Task").strip()
            description = str(task.get("description") or "").strip()
            formatted.append(f"{title}: {description}" if description else title)
        elif isinstance(task, str) and task.strip():
            formatted.append(task.strip())
    return formatted


def run_agent(request_text: str) -> Dict[str, Any]:
    doc_type = detect_document_type(request_text)
    plan_data = build_plan(request_text, doc_type)
    resolved_doc_type = plan_data.get("document_type", doc_type)
    tasks = plan_data.get("tasks", [])
    plan = _format_tasks(tasks)
    assumptions = plan_data.get("assumptions", [])
    summary = plan_data.get("summary", f"Planned the work for a {resolved_doc_type.lower()}.")
    planning_mode = plan_data.get("source", "heuristic")

    execution_context = execute_document_generation(request_text, resolved_doc_type, tasks, assumptions)
    reflected = reflect_on_output(execution_context)
    improved_context = reflected.get("execution_context", execution_context)
    document_path = build_document_from_context(improved_context, reflected.get("assumptions", assumptions))
    return {
        "message": f"Completed the request and generated a {resolved_doc_type.lower()} document.",
        "plan": reflected.get("plan", plan),
        "assumptions": reflected.get("assumptions", assumptions),
        "document_name": document_path.name,
        "document_path": f"/download/{document_path.name}",
        "summary": reflected.get("summary", summary),
        "planning_mode": planning_mode,
        "tasks": tasks,
        "document_type": resolved_doc_type,
    }
