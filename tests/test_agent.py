"""Tests for LangGraph agent."""

import pytest

from src.agent.nodes import parse_query, generate_response


@pytest.mark.asyncio
async def test_parse_query_segmentation():
    state = {"query": "Segment the land cover in this image", "image_path": "test.png"}
    result = await parse_query(state)
    assert "segmentation" in result["intent"]
    assert result["has_image"] is True


@pytest.mark.asyncio
async def test_parse_query_detection():
    state = {"query": "Detect buildings in the satellite image", "image_path": "test.png"}
    result = await parse_query(state)
    assert "detection" in result["intent"]


@pytest.mark.asyncio
async def test_parse_query_ndvi():
    state = {"query": "What is the vegetation health?", "image_path": "test.tif"}
    result = await parse_query(state)
    assert "ndvi" in result["intent"]


@pytest.mark.asyncio
async def test_parse_query_no_image():
    state = {"query": "Analyze this area", "image_path": None}
    result = await parse_query(state)
    assert result["has_image"] is False


@pytest.mark.asyncio
async def test_generate_response_no_image():
    state = {
        "query": "What can you do?",
        "has_image": False,
        "analysis": None,
    }
    result = await generate_response(state)
    assert "provide an image" in result["response"].lower()


@pytest.mark.asyncio
async def test_build_agent_graph():
    from src.agent.graph import build_agent_graph

    graph = build_agent_graph()
    assert graph is not None
