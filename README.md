# slack-indexed

MCP server for semantic search over indexed Slack channels stored in Qdrant, with Cohere reranking via AWS Bedrock.

## Prerequisites

- A running Qdrant instance with an indexed Slack collection (created by [alaan-slack-index-mcp](https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp))
- AWS credentials with access to Bedrock (Titan embeddings + Cohere rerank)

## Tools

| Tool | Description |
|------|-------------|
| `search` | Semantic search with Cohere reranking, filterable by source type, channel, and user |
| `get_thread` | Retrieve all chunks for a specific Slack thread including linked resources |
| `list_channels` | List indexed channels with document counts |
| `collection_stats` | Summary statistics (documents, threads, sources, channels) |
| `list_users` | List all known participant names for filtering |

## Usage

### Direct with uvx (no install)

Using an AWS profile:
```bash
uvx --from git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool slack-indexed --profile my-profile --qdrant-url http://localhost:6333
```

Using explicit AWS credentials and a Qdrant API key:
```bash
uvx --from git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool slack-indexed \
  --aws-access-key AKIA... \
  --aws-secret-key wJal... \
  --qdrant-url http://your-qdrant-host:6333 \
  --qdrant-api-key your-qdrant-api-key
```

### Local development

```bash
uv sync
uv run slack-indexed --profile my-profile --qdrant-url http://localhost:6333
```

### Claude Code (`~/.claude.json`)

With AWS profile:
```json
{
  "mcpServers": {
    "slack-indexed": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool",
        "slack-indexed",
        "--profile",
        "my-profile",
        "--qdrant-url",
        "http://localhost:6333",
        "--qdrant-api-key",
        "xxxx-api-key-xxxx"
      ]
    }
  }
}
```

With explicit AWS credentials:
```json
{
  "mcpServers": {
    "slack-indexed": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool",
        "slack-indexed",
        "--aws-access-key",
        "AKIA...",
        "--aws-secret-key",
        "wJal...",
        "--qdrant-url",
        "http://your-qdrant-host:6333",
        "--qdrant-api-key",
        "xxxx-api-key-xxxx"
      ]
    }
  }
}
```

### Cursor (`.cursor/mcp.json`)

With AWS profile:
```json
{
  "mcpServers": {
    "slack-indexed": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool",
        "slack-indexed",
        "--profile",
        "my-profile",
        "--qdrant-api-key",
        "xxxx-api-key-xxxx"
      ]
    }
  }
}
```

With explicit AWS credentials:
```json
{
  "mcpServers": {
    "slack-indexed": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/KanvaBhatia-Alaan/alaan-slack-index-mcp-tool",
        "slack-indexed",
        "--aws-access-key",
        "AKIA...",
        "--aws-secret-key",
        "wJal...",
        "--qdrant-url",
        "http://your-qdrant-host:6333",
        "--qdrant-api-key",
        "xxxx-api-key-xxxx"
      ]
    }
  }
}
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--qdrant-url` | `http://localhost:6333` | Qdrant server URL |
| `--collection` | `slack_index` | Qdrant collection name |
| `--profile` | _(env default)_ | AWS profile name |
| `--region` | `us-east-1` | AWS region for Bedrock |
| `--aws-access-key` | _(env default)_ | AWS access key ID (use instead of `--profile`) |
| `--aws-secret-key` | _(env default)_ | AWS secret access key (use with `--aws-access-key`) |
| `--qdrant-api-key` | _(none)_ | Qdrant API key for authenticated access |
| `--qdrant-timeout` | `30` | Qdrant request timeout in seconds |
