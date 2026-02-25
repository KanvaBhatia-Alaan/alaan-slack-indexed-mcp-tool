from __future__ import annotations

from typing import Any, Dict, List, Optional

import cohere


class Reranker:
    def __init__(
        self,
        region: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: str = "cohere.rerank-v3-5:0",
    ):
        self.client = cohere.BedrockClientV2(
            aws_region=region,
            aws_access_key=access_key,
            aws_secret_key=secret_key,
        )
        self.model = model

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        response = self.client.rerank(
            model=self.model, query=query, documents=documents, top_n=top_n
        )
        return [
            {"index": r.index, "score": r.relevance_score} for r in response.results
        ]
