"""
AWS Bedrock KB Smart Tool — MCP server (SSE/HTTP)

This server exposes a single MCP tool `search_knowledge_base` that mirrors the logic of your
Flowise node:
  • LLM-based TOP-N subfolder routing (with MCP sampling) + heuristic fallback
  • Build a metadata RetrievalFilter over `x-amz-bedrock-kb-source-uri` for S3 prefixes
  • Query Amazon Bedrock Knowledge Bases via the Retrieve API
  • Return ranked chunks with source, score, and text

Transport: HTTP + Server-Sent Events (SSE) via FastMCP.
The MCP endpoint will be available at: http://HOST:PORT/mcp

Quick start
-----------
1) Install deps:
   pip install fastmcp boto3 pydantic python-dotenv

2) Set environment variables (or create a .env file):
   AWS_REGION=us-east-1
   KB_ID=DRXXU5RCGD
   # JSON array of objects: [{"key":"bongda/2001/","name":"2001"}, ...]
   SUBFOLDERS_JSON=[{"key":"bongda/2001/","name":"2001"},{"key":"bongda/2002/","name":"World Cup 2002"}]
   MAX_SUBFOLDERS=3
   TOP_K=5
   SEARCH_TYPE=HYBRID     # or SEMANTIC
   USE_LLM_FILTER=true    # uses MCP sampling; falls back to heuristic if not available
   READ_GROUP_PRIVATE=false
   READ_USER_PRIVATE=false
   GROUP_ID=
   USER_ID=
   PORT=8000

3) Run:
   python aws_kb_smart_tool_mcp.py
   # Endpoint: http://localhost:8000/mcp (SSE-enabled)

Notes
-----
• LLM routing is done with MCP Sampling (asks the client/host LLM); no provider API keys needed here.
• If sampling is not available or times out, a deterministic heuristic selects the top-N subfolders.
• RetrievalFilter uses `stringContains` / `startsWith` on `x-amz-bedrock-kb-source-uri` with an OR of per-folder filters,
  plus optional group/user private paths: <root>/groups/<GROUP_ID>/ and <root>/users/<USER_ID>/.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import boto3
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import SamplingMessage, TextContent

# ------------------------------------------------------------
# Config & models
# ------------------------------------------------------------

load_dotenv()  # optional .env support

@dataclass
class Subfolder:
    key: str
    name: str


class SearchArgs(BaseModel):
    query: str = Field(..., description="User query to search the knowledge base")
    max_subfolders: int = Field(default=int(os.getenv("MAX_SUBFOLDERS", 3)), ge=1, le=10,
                                description="Maximum number of subfolders to route into")
    top_k: int = Field(default=int(os.getenv("TOP_K", 5)), ge=1, le=50,
                       description="Number of results to return from Bedrock")
    search_type: str = Field(default=os.getenv("SEARCH_TYPE", "HYBRID"), regex=r"^(HYBRID|SEMANTIC)$",
                             description="Bedrock overrideSearchType: HYBRID or SEMANTIC")
    use_llm_filter: bool = Field(default=os.getenv("USE_LLM_FILTER", "true").lower() == "true",
                                 description="If true, try LLM routing via MCP sampling")
    include_scores: bool = Field(default=True, description="Return similarity scores when available")
    read_group_private: bool = Field(default=os.getenv("READ_GROUP_PRIVATE", "false").lower() == "true")
    read_user_private: bool = Field(default=os.getenv("READ_USER_PRIVATE", "false").lower() == "true")
    group_id: Optional[str] = Field(default=os.getenv("GROUP_ID") or None)
    user_id: Optional[str] = Field(default=os.getenv("USER_ID") or None)


# Global config from env
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KB_ID = os.getenv("KB_ID")
PORT = int(os.getenv("PORT", 8000))

# Parse subfolders from env or sidecar JSON file
_SUBFOLDERS_JSON = os.getenv("SUBFOLDERS_JSON", "")
if not _SUBFOLDERS_JSON and os.path.exists("subfolders.json"):
    _SUBFOLDERS_JSON = open("subfolders.json", "r", encoding="utf-8").read()

try:
    SUBFOLDERS: List[Subfolder] = [Subfolder(**item) for item in json.loads(_SUBFOLDERS_JSON)] if _SUBFOLDERS_JSON else []
except Exception as e:
    raise RuntimeError(f"Invalid SUBFOLDERS_JSON. Expecting a JSON array of objects with 'key' and 'name'. Error: {e}")

if not KB_ID:
    raise RuntimeError("KB_ID environment variable is required")

# Initialize Bedrock Agent Runtime client (credentials are taken from env/role)
bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

# Initialize MCP server
mcp = FastMCP(name="aws-kb-smart-tool", website_url="https://aws.amazon.com/bedrock/knowledge-bases/")

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

async def llm_pick_subfolders_via_sampling(
    ctx: Context[ServerSession, None], query: str, subfolders: List[Subfolder], top_n: int, timeout_s: float = 20.0
) -> List[Subfolder]:
    """Ask the host LLM (via MCP Sampling) to pick TOP-N subfolders. Returns [] on failure/timeouts."""
    if not subfolders:
        return []

    options = "\n".join(f"{i+1}. {sf.name}" for i, sf in enumerate(subfolders))
    prompt = (
        f"Analyze this query and select the TOP {top_n} most relevant subfolders.\n\n"
        f"Query: \"{query}\"\n\nAvailable subfolders:\n{options}\n\n"
        f"Instructions:\n- Return EXACTLY {top_n} numbers (comma-separated)\n"
        f"- Order by relevance (most relevant first)\n- Use numbers 1-{len(subfolders)}\n- If query relates to multiple topics, select diverse folders\n"
        f"- Example format: \"3, 7, 12\"\n\nYour answer (numbers only):"
    )

    async def _sample() -> str:
        result = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            max_tokens=32,
            temperature=0,
        )
        if result.content.type == "text":
            return result.content.text.strip()
        return ""

    try:
        text = await asyncio.wait_for(_sample(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return []
    except Exception:
        return []

    idxs = []
    for tok in json.dumps(text):  # simple scan for digits
        pass
    # Better: regex for numbers
    import re
    nums = re.findall(r"\d+", text)
    for n in nums:
        i = int(n) - 1
        if 0 <= i < len(subfolders):
            idxs.append(i)
        if len(idxs) >= top_n:
            break

    seen = set()
    ordered_unique = [i for i in idxs if not (i in seen or seen.add(i))]
    return [subfolders[i] for i in ordered_unique[:top_n]]


def heuristic_match(query: str, subfolders: List[Subfolder], top_n: int) -> List[Subfolder]:
    q = (query or "").lower()
    if not subfolders:
        return []
    scores: List[Tuple[Subfolder, float]] = []
    q_words = [w for w in q.split() if w]

    for sf in subfolders:
        name = sf.name.lower()
        score = 0.0
        if q and name in q:
            score += 10
        if q and q in name:
            score += 8
        name_words = [w for w in name.split() if w]
        overlap = 0
        for w in q_words:
            for nw in name_words:
                if w in nw or nw in w:
                    overlap += 1
                    break
        score += overlap * 3
        scores.append((sf, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sf for sf, _ in scores[:top_n]]


def build_filter(
    selected: List[Subfolder], read_group_private: bool, read_user_private: bool, group_id: Optional[str], user_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Builds an OR filter on `x-amz-bedrock-kb-source-uri` for selected folders,
    and optionally adds group/user paths per root folder.
    Returns a RetrievalFilter dict compatible with boto3 `retrieve`.
    """
    if not selected:
        return None

    or_all: List[Dict[str, Any]] = []

    for sf in selected:
        # Prefer startsWith if your store supports it; fallback to stringContains
        or_all.append({
            "startsWith": {"key": "x-amz-bedrock-kb-source-uri", "value": sf.key}
        })
        or_all.append({
            "stringContains": {"key": "x-amz-bedrock-kb-source-uri", "value": sf.key}
        })

        root = sf.key.split("/")[0] if "/" in sf.key else sf.key
        if read_group_private and group_id and root:
            group_path = f"{root}/groups/{group_id}/"
            or_all.append({
                "startsWith": {"key": "x-amz-bedrock-kb-source-uri", "value": group_path}
            })
            or_all.append({
                "stringContains": {"key": "x-amz-bedrock-kb-source-uri", "value": group_path}
            })
        if read_user_private and user_id and root:
            user_path = f"{root}/users/{user_id}/"
            or_all.append({
                "startsWith": {"key": "x-amz-bedrock-kb-source-uri", "value": user_path}
            })
            or_all.append({
                "stringContains": {"key": "x-amz-bedrock-kb-source-uri", "value": user_path}
            })

    if len(or_all) == 1:
        return or_all[0]
    return {"orAll": or_all}


def format_retrieval_results(resp: Dict[str, Any], include_scores: bool = True) -> Dict[str, Any]:
    results = []
    for i, item in enumerate(resp.get("retrievalResults", []) or []):
        content = item.get("content", {})
        text = content.get("text") or content.get("byteContent") or ""
        location = item.get("location")
        metadata = item.get("metadata", {})
        score = item.get("score") if include_scores else None
        results.append({
            "rank": i + 1,
            "score": float(score) if score is not None else None,
            "location": location,
            "metadata": metadata,
            "text": text,
        })
    return {"count": len(results), "items": results}


# ------------------------------------------------------------
# MCP Tool
# ------------------------------------------------------------

@mcp.tool()
async def search_knowledge_base(args: SearchArgs, ctx: Context[ServerSession, None]) -> Dict[str, Any]:
    """Search Bedrock Knowledge Base with intelligent TOP-N subfolder routing.

    Args:
        args: SearchArgs (query, max_subfolders, top_k, search_type, etc.)
    Returns:
        JSON dict with routing decisions and retrieval results.
    """
    start_total = time.time()
    await ctx.info(f"Query: {args.query}")

    # Step 1: choose subfolders
    chosen: List[Subfolder] = []
    if SUBFOLDERS:
        if args.use_llm_filter:
            await ctx.debug("Trying LLM (MCP sampling) to route subfolders…")
            chosen = await llm_pick_subfolders_via_sampling(ctx, args.query, SUBFOLDERS, args.max_subfolders)
        if not chosen:
            await ctx.debug("LLM routing unavailable or failed — using heuristic matching")
            chosen = heuristic_match(args.query, SUBFOLDERS, args.max_subfolders)
    else:
        await ctx.warning("No subfolders configured — searching entire KB (may be expensive)")

    # Step 2: build filter
    retrieval_filter = build_filter(chosen, args.read_group_private, args.read_user_private, args.group_id, args.user_id)

    # Step 3: retrieve
    await ctx.debug("Calling Bedrock Retrieve API…")
    start_retrieve = time.time()
    retrieve_kwargs: Dict[str, Any] = {
        "knowledgeBaseId": KB_ID,
        "retrievalQuery": {"text": args.query},
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": args.top_k,
                "overrideSearchType": args.search_type,
            }
        },
    }
    if retrieval_filter:
        retrieve_kwargs["filter"] = retrieval_filter

    resp = bedrock_runtime.retrieve(**retrieve_kwargs)
    took_ms = int((time.time() - start_retrieve) * 1000)

    # Step 4: format
    formatted = format_retrieval_results(resp, include_scores=args.include_scores)

    total_ms = int((time.time() - start_total) * 1000)

    return {
        "selected_subfolders": [sf.__dict__ for sf in chosen],
        "filter": retrieval_filter,
        "retrieve_took_ms": took_ms,
        "total_took_ms": total_ms,
        "results": formatted,
    }


# ------------------------------------------------------------
# HTTP/SSE hosting
# ------------------------------------------------------------

if __name__ == "__main__":
    # Start an HTTP server with SSE-enabled MCP endpoint at /mcp
    # You can also: mcp.run(transport="stdio") for stdin/stdout transport
    port=int(os.getenv("PORT",8000))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
