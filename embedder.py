from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024
BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


class BedrockEmbedder:
    def __init__(
        self,
        region: str,
        profile: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        session_kw: Dict[str, Any] = {"region_name": region}
        if profile:
            session_kw["profile_name"] = profile
        if access_key and secret_key:
            session_kw["aws_access_key_id"] = access_key
            session_kw["aws_secret_access_key"] = secret_key
        self.client = boto3.Session(**session_kw).client("bedrock-runtime")

    def _embed_single(self, text: str) -> List[float]:
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": EMBEDDING_DIM,
                "embeddingTypes": ["float"],
            }
        )
        response = self.client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(text) for text in texts]
