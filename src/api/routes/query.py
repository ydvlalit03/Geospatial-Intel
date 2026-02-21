"""Natural language query endpoint using LangGraph agent."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import QueryRequest, QueryResponse

router = APIRouter()

# Will be set during app startup
agent_graph = None


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Send a natural language query to the geospatial AI agent."""
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await agent_graph.ainvoke(
            {
                "query": request.query,
                "image_path": request.image_path,
            }
        )

        return QueryResponse(
            query=request.query,
            response=result.get("response", "No response generated."),
            analysis=result.get("analysis"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
