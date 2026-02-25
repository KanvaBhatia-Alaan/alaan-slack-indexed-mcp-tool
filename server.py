"""MCP server exposing semantic search over an indexed Slack collection in Qdrant.

Tools provided:
- search: Semantic search with Cohere reranking across Slack threads and linked resources
- get_thread: Retrieve all chunks for a specific Slack thread
- list_channels: List indexed channels with document counts
- collection_stats: Summary statistics about the indexed collection
- list_users: List all known participants across indexed threads

Run with uvx and pass Qdrant URL (and optional AWS args) as CLI args, e.g.:
  uvx --from git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool slack-indexed
  uvx --from git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool slack-indexed --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from embedder import BedrockEmbedder
from reranker import Reranker
from vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-init state (populated by main() from CLI args)
# ---------------------------------------------------------------------------
_qdrant_url: str = "http://localhost:6333"
_qdrant_collection: str = "slack_index"
_aws_profile: Optional[str] = None
_aws_region: str = "us-east-1"
_aws_access_key: Optional[str] = None
_aws_secret_key: Optional[str] = None
_qdrant_timeout: int = 30
_qdrant_api_key: Optional[str] = None
_embedder: Any = None
_store: Any = None
_reranker: Any = None

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "slack-indexed",
    instructions="Search indexed Slack channel messages and linked resources (GitHub, Linear, Notion).",
)

SOURCE_TYPES = {
    "slack_thread",
    "github_issue",
    "github_pr",
    "github_file",
    "linear_issue",
    "notion_page",
}


# ---------------------------------------------------------------------------
# Lazy component accessors
# ---------------------------------------------------------------------------
def _get_embedder() -> BedrockEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BedrockEmbedder(
            region=_aws_region,
            profile=_aws_profile,
            access_key=_aws_access_key,
            secret_key=_aws_secret_key,
        )
    return _embedder


def _get_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore(
            url=_qdrant_url,
            collection_name=_qdrant_collection,
            embedder=_get_embedder(),
            timeout=_qdrant_timeout,
            api_key=_qdrant_api_key,
        )
    return _store


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(
            region=_aws_region,
            access_key=_aws_access_key,
            secret_key=_aws_secret_key,
        )
    return _reranker


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def search(
    query: str,
    n_results: int = 10,
    source_filter: str | None = None,
    channel_filter: str | None = None,
    user_filter: str | None = None,
    include_links: bool = True,
) -> dict:
    """Search the Slack index using semantic similarity with Cohere reranking.

    Args:
        query: Natural language search query.
        n_results: Number of results to return (default 10, max 50).
        source_filter: Filter by source type. One of: slack_thread, github_issue,
                       github_pr, github_file, linear_issue, notion_page.
        channel_filter: Filter by Slack channel name (partial match).
        user_filter: Filter by participant name (partial match on thread_users).
        include_links: If False, exclude linked resources and return only Slack threads.
    """
    store = _get_store()
    reranker = _get_reranker()
    n_results = min(max(1, n_results), 50)

    where_clauses: List[Dict] = []
    if source_filter:
        if source_filter not in SOURCE_TYPES:
            return {
                "error": f"Invalid source_filter. Must be one of: {sorted(SOURCE_TYPES)}"
            }
        where_clauses.append({"source": source_filter})
    elif not include_links:
        where_clauses.append({"source": "slack_thread"})

    if channel_filter:
        where_clauses.append({"channel_name": {"$contains": channel_filter}})
    if user_filter:
        where_clauses.append({"thread_users": {"$contains": user_filter}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    fetch_n = n_results * 3
    kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": fetch_n}
    if where:
        kwargs["where"] = where

    try:
        results = store.query(**kwargs)
        candidates_ids = results["ids"][0]
        candidates_docs = results["documents"][0]
        candidates_distances = results["distances"][0]
        candidates_metas = results["metadatas"][0]

        if not candidates_docs:
            return {"query": query, "count": 0, "results": []}

        try:
            rerank_results = reranker.rerank(
                query=query, documents=candidates_docs, top_n=n_results
            )
            formatted = []
            for r in rerank_results:
                idx = r["index"]
                formatted.append(
                    {
                        "id": candidates_ids[idx],
                        "text": candidates_docs[idx],
                        "rerank_score": round(r["score"], 4),
                        "vector_distance": round(candidates_distances[idx], 4),
                        "metadata": candidates_metas[idx],
                    }
                )
        except Exception as e:
            logger.warning(f"Reranker failed, falling back to vector results: {e}")
            formatted = []
            for i in range(min(n_results, len(candidates_docs))):
                formatted.append(
                    {
                        "id": candidates_ids[i],
                        "text": candidates_docs[i],
                        "rerank_score": None,
                        "vector_distance": round(candidates_distances[i], 4),
                        "metadata": candidates_metas[i],
                    }
                )

        return {"query": query, "count": len(formatted), "results": formatted}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_thread(thread_ts: str, channel_id: str | None = None) -> dict:
    """Retrieve all chunks for a specific Slack thread, including any linked resources.

    Use this after search() to get the full thread context.

    Args:
        thread_ts: The thread timestamp (from search result metadata).
        channel_id: Optional channel ID to narrow the lookup.
    """
    store = _get_store()
    where: Dict[str, Any] = {"thread_ts": thread_ts}
    if channel_id:
        where = {"$and": [{"thread_ts": thread_ts}, {"channel_id": channel_id}]}

    results = store.get(where=where, include=["documents", "metadatas"])

    thread_chunks = []
    link_chunks = []

    for doc_id, text, meta in zip(
        results["ids"], results["documents"], results["metadatas"]
    ):
        entry = {"id": doc_id, "text": text, "metadata": meta}
        if meta.get("source") == "slack_thread":
            thread_chunks.append(entry)
        else:
            link_chunks.append(entry)

    thread_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
    link_chunks.sort(
        key=lambda x: (
            x["metadata"].get("url", ""),
            x["metadata"].get("chunk_index", 0),
        )
    )

    return {
        "thread_ts": thread_ts,
        "thread_chunks": thread_chunks,
        "linked_resources": link_chunks,
    }


@mcp.tool()
def list_channels() -> dict:
    """List all indexed channels and their document counts."""
    store = _get_store()
    all_docs = store.get(include=["metadatas"])

    channels: Dict[str, Dict] = {}
    for meta in all_docs["metadatas"]:
        name = meta.get("channel_name", "unknown")
        if name not in channels:
            channels[name] = {
                "channel_id": meta.get("channel_id", ""),
                "total_docs": 0,
                "threads": 0,
                "links": 0,
            }
        channels[name]["total_docs"] += 1
        if meta.get("source") == "slack_thread":
            channels[name]["threads"] += 1
        else:
            channels[name]["links"] += 1

    return {"channels": channels, "total_documents": store.count()}


@mcp.tool()
def collection_stats() -> dict:
    """Get summary statistics about the indexed collection."""
    store = _get_store()
    all_docs = store.get(include=["metadatas"])

    source_counts: Dict[str, int] = {}
    channel_counts: Dict[str, int] = {}
    unique_threads: set[str] = set()

    for meta in all_docs["metadatas"]:
        source = meta.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        channel = meta.get("channel_name", "unknown")
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        ts = meta.get("thread_ts")
        if ts:
            unique_threads.add(ts)

    return {
        "total_documents": store.count(),
        "unique_threads": len(unique_threads),
        "by_source": source_counts,
        "by_channel": channel_counts,
    }


@mcp.tool()
def list_users() -> dict:
    """List all known participants across indexed Slack threads.

    Scans the thread_users metadata field to return all unique participant names.
    Useful for discovering valid values for the user_filter parameter in search().
    """
    store = _get_store()
    all_docs = store.get(include=["metadatas"])

    users: set[str] = set()
    for meta in all_docs["metadatas"]:
        thread_users = meta.get("thread_users", "")
        if thread_users:
            for name in thread_users.split(", "):
                name = name.strip()
                if name:
                    users.add(name)

    return {"users": sorted(users), "count": len(users)}


# ---------------------------------------------------------------------------
# CLI & entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MCP server for searching indexed Slack data in Qdrant. "
            "Pass AWS credentials and Qdrant connection details."
        )
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="slack_index",
        help="Qdrant collection name (default: slack_index)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile name (e.g. from ~/.aws/credentials). Omit to use default profile / env.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)",
    )
    parser.add_argument(
        "--aws-access-key",
        type=str,
        default=None,
        help="AWS access key ID (omit to use profile / env)",
    )
    parser.add_argument(
        "--aws-secret-key",
        type=str,
        default=None,
        help="AWS secret access key (omit to use profile / env)",
    )
    parser.add_argument(
        "--qdrant-timeout",
        type=int,
        default=30,
        help="Qdrant request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (omit to use no API key)",
    )   
    return parser.parse_known_args()[0]


def main() -> None:
    """Entry point for uvx / MCP: parse args and run the server over stdio."""
    global _qdrant_url, _qdrant_collection, _aws_profile, _aws_region
    global _aws_access_key, _aws_secret_key, _qdrant_timeout
    global _qdrant_api_key
    args = _parse_args()
    _qdrant_url = args.qdrant_url
    _qdrant_collection = args.collection
    _aws_profile = args.profile
    _aws_region = args.region
    _aws_access_key = args.aws_access_key
    _aws_secret_key = args.aws_secret_key
    _qdrant_timeout = args.qdrant_timeout
    _qdrant_api_key = args.qdrant_api_key
    logger.info(
        f"Starting slack-indexed MCP server (qdrant={_qdrant_url}, "
        f"collection={_qdrant_collection}, region={_aws_region}), api key set = {_qdrant_api_key is not None}"
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
