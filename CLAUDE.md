# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An MCP (Model Context Protocol) server that exposes semantic search over Slack channels indexed in Qdrant, with Cohere reranking via AWS Bedrock. Read-only — no write operations.

## Commands

```bash
# Install dependencies
uv sync

# Run locally (dev)
uv run slack-indexed --qdrant-url http://localhost:6333 --profile my-profile

# Run with explicit AWS creds + Qdrant API key
uv run slack-indexed \
  --aws-access-key AKIA... --aws-secret-key wJal... \
  --qdrant-url http://host:6333 --qdrant-api-key xxx

# Run via uvx (no install)
uvx --from git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool slack-indexed
```

No test suite exists. Manual testing requires a running Qdrant instance with pre-indexed data and AWS Bedrock access.

## Architecture

```
server.py  →  Entry point, CLI arg parsing, 5 MCP tools (@mcp.tool())
embedder.py  →  Amazon Titan v2 embeddings (1024-dim) via boto3/Bedrock
vector_store.py  →  Qdrant client wrapper with custom filter translation
reranker.py  →  Cohere rerank v3.5 via AWS Bedrock
```

**Key patterns:**

- **Lazy initialization**: Global state (`_embedder`, `_store`, `_reranker`) is `None` at startup. Accessor functions (`_get_embedder()`, `_get_store()`, `_get_reranker()`) instantiate on first tool call, after CLI args have been parsed in `main()`.
- **Graceful degradation**: If the Cohere reranker fails, search falls back to raw vector similarity results.
- **Filter translation**: `vector_store._translate_filter()` converts a simple dict DSL (`{"field": value}`, `{"field": {"$contains": text}}`, `{"$and": [...]}`) into Qdrant `FieldCondition`/`Filter` objects.
- **Transport**: Runs over stdio (`mcp.run(transport="stdio")`). Logs go to stderr and are not visible to MCP clients.

## MCP Tools

All tools are stateless and read-only:
- `search` — semantic search with reranking, filterable by source type / channel / user
- `get_thread` — full thread + linked resources by thread_ts
- `list_channels`, `collection_stats`, `list_users` — inventory/metadata queries

Source types: `slack_thread`, `github_issue`, `github_pr`, `github_file`, `linear_issue`, `notion_page`.

## Build & Distribution

Uses **hatchling** as the PEP 517 build backend. The 4 Python files are force-included via `[tool.hatch.build.targets.wheel]` since there's no package directory. Entry point: `slack-indexed = "server:main"`.
