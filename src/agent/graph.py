"""LangGraph agent definition for geospatial intelligence queries."""

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes import execute_ml, generate_response, parse_query


class AgentState(dict):
    """State schema for the geospatial agent graph."""

    query: str
    image_path: str | None
    intent: list[str]
    has_image: bool
    analysis: dict | None
    response: str


def should_execute(state: dict[str, Any]) -> str:
    """Route to ML execution if an image is provided, else skip to response."""
    if state.get("has_image") and state.get("intent"):
        return "execute"
    return "respond"


def build_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent for geospatial queries.

    Flow:
        parse_query -> (has image?) -> execute_ml -> generate_response
                                   |-> generate_response (no image)
    """
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("parse", parse_query)
    graph.add_node("execute", execute_ml)
    graph.add_node("respond", generate_response)

    # Add edges
    graph.set_entry_point("parse")
    graph.add_conditional_edges(
        "parse",
        should_execute,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
