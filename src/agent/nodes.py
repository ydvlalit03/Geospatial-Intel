"""Agent node functions for the LangGraph state graph."""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a geospatial intelligence analyst AI. You help users analyze
satellite imagery by running segmentation, object detection, and vegetation analysis.

You have access to the following tools:
- run_segmentation: Analyze land cover types (buildings, roads, vegetation, water, barren)
- run_detection: Detect objects (vehicles, buildings, ships, aircraft, etc.)
- compute_vegetation_index: Calculate NDVI vegetation health index

When a user asks about an image, determine which tool(s) to use and explain the results
in clear, actionable language. If no image is provided, explain what analysis you could
perform if given one."""


async def parse_query(state: dict[str, Any]) -> dict[str, Any]:
    """Parse the user's natural language query to determine intent."""
    query = state["query"]
    image_path = state.get("image_path")

    query_lower = query.lower()

    # Determine which tools to call
    tools_to_run = []

    if any(w in query_lower for w in ["segment", "land cover", "land use", "classify"]):
        tools_to_run.append("segmentation")
    if any(w in query_lower for w in ["detect", "find", "locate", "object", "count"]):
        tools_to_run.append("detection")
    if any(w in query_lower for w in ["vegetation", "ndvi", "green", "health", "crop"]):
        tools_to_run.append("ndvi")

    # Default: run segmentation if no specific intent detected
    if not tools_to_run and image_path:
        tools_to_run.append("segmentation")

    return {
        **state,
        "intent": tools_to_run,
        "has_image": image_path is not None,
    }


async def execute_ml(state: dict[str, Any]) -> dict[str, Any]:
    """Execute ML tools based on parsed intent."""
    intent = state.get("intent", [])
    image_path = state.get("image_path")
    analysis = {}

    if not image_path:
        return {**state, "analysis": None}

    from src.agent.tools import (
        compute_vegetation_index,
        run_detection,
        run_segmentation,
    )

    for task in intent:
        try:
            if task == "segmentation":
                analysis["segmentation"] = run_segmentation.invoke(
                    {"image_path": image_path}
                )
            elif task == "detection":
                analysis["detection"] = run_detection.invoke(
                    {"image_path": image_path}
                )
            elif task == "ndvi":
                analysis["ndvi"] = compute_vegetation_index.invoke(
                    {"image_path": image_path}
                )
        except Exception as e:
            analysis[task] = {"error": str(e)}

    return {**state, "analysis": analysis}


async def generate_response(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a natural language response summarizing the analysis."""
    query = state["query"]
    analysis = state.get("analysis")
    has_image = state.get("has_image", False)

    if not has_image:
        response = (
            "I can analyze satellite imagery for you. Please provide an image path "
            "along with your query. I can perform:\n"
            "- **Land cover segmentation** (buildings, roads, vegetation, water)\n"
            "- **Object detection** (vehicles, structures, ships)\n"
            "- **Vegetation health analysis** (NDVI index)\n\n"
            f"Your query: \"{query}\""
        )
        return {**state, "response": response}

    if not analysis:
        return {**state, "response": "Analysis could not be completed."}

    # Build response from analysis results
    parts = [f"**Analysis for your query:** \"{query}\"\n"]

    if "segmentation" in analysis:
        seg = analysis["segmentation"]
        if "error" in seg:
            parts.append(f"Segmentation error: {seg['error']}")
        else:
            parts.append("**Land Cover Analysis:**")
            for cls_name, pct in seg.get("class_distribution", {}).items():
                parts.append(f"  - {cls_name}: {pct*100:.1f}%")

    if "detection" in analysis:
        det = analysis["detection"]
        if "error" in det:
            parts.append(f"Detection error: {det['error']}")
        else:
            parts.append(f"\n**Object Detection:** {det.get('num_detections', 0)} objects found")
            for cls_name, count in det.get("class_summary", {}).items():
                parts.append(f"  - {cls_name}: {count}")

    if "ndvi" in analysis:
        ndvi = analysis["ndvi"]
        if "error" in ndvi:
            parts.append(f"NDVI error: {ndvi['error']}")
        else:
            parts.append(f"\n**Vegetation Health (NDVI):**")
            parts.append(f"  - Mean NDVI: {ndvi.get('mean_ndvi', 0):.3f}")
            parts.append(f"  - Vegetation coverage: {ndvi.get('vegetation_coverage', 0)*100:.1f}%")

    return {**state, "response": "\n".join(parts)}
