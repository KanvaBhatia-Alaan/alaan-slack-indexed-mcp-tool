from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    VectorParams,
)

from embedder import EMBEDDING_DIM, BedrockEmbedder

logger = logging.getLogger(__name__)


def _string_id_to_uuid(string_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, string_id))


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, embedder: BedrockEmbedder, timeout: int = 30, api_key: Optional[str] = None):
        logger.info(f"Initializing QdrantVectorStore with url: {url}, collection_name: {collection_name}")
        host = url.split("//")[1]
        port = 443 if "https" in url else 6333
        https = True if "https" in url else False
        self.client = QdrantClient(host=host, port=port, timeout=timeout, api_key=api_key, https= https)
        self.embedder = embedder
        self.collection_name = collection_name
        self._ensure_collection()
        logger.info(f"Qdrant collection '{self.collection_name}' ready at {url}")

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )

    def query(
        self, query_texts: List[str], n_results: int = 10, **kwargs: Any
    ) -> Dict[str, Any]:
        query_embeddings = self.embedder.embed_queries(query_texts)

        all_ids: List[List[str]] = []
        all_docs: List[List[str]] = []
        all_distances: List[List[float]] = []
        all_metadatas: List[List[Dict]] = []

        qdrant_filter = None
        if "where" in kwargs:
            qdrant_filter = _translate_filter(kwargs["where"])

        for query_embedding in query_embeddings:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=n_results,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points

            ids, docs, distances, metadatas = [], [], [], []
            for point in results:
                ids.append(point.payload.get("_string_id", str(point.id)))
                docs.append(point.payload.get("document", ""))
                distances.append(
                    1.0 - point.score if point.score is not None else 1.0
                )
                meta = {
                    k: v
                    for k, v in point.payload.items()
                    if k not in ("document", "_string_id")
                }
                metadatas.append(meta)

            all_ids.append(ids)
            all_docs.append(docs)
            all_distances.append(distances)
            all_metadatas.append(metadatas)

        return {
            "ids": all_ids,
            "documents": all_docs,
            "distances": all_distances,
            "metadatas": all_metadatas,
        }

    def get(
        self, where: Optional[Dict] = None, include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        qdrant_filter = _translate_filter(where) if where else None
        include = include or []

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for point in results:
                ids.append(point.payload.get("_string_id", str(point.id)))
                if "documents" in include:
                    documents.append(point.payload.get("document", ""))
                meta = {
                    k: v
                    for k, v in point.payload.items()
                    if k not in ("document", "_string_id")
                }
                if "metadatas" in include:
                    metadatas.append(meta)
            if next_offset is None:
                break
            offset = next_offset

        result: Dict[str, Any] = {"ids": ids}
        if "documents" in include:
            result["documents"] = documents
        if "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count


def _translate_filter(where: Dict) -> Filter:
    if "$and" in where:
        conditions = [_translate_condition(c) for c in where["$and"]]
        return Filter(must=conditions)
    cond = _translate_condition(where)
    return Filter(must=[cond])


def _translate_condition(condition: Dict) -> FieldCondition:
    non_dollar_keys = [k for k in condition if not k.startswith("$")]
    if len(non_dollar_keys) > 1:
        raise ValueError(
            f"Filter condition must have exactly one field key, got {non_dollar_keys}. "
            f"Use $and to combine multiple conditions."
        )
    for key, value in condition.items():
        if key.startswith("$"):
            continue
        if isinstance(value, dict):
            if "$contains" in value:
                return FieldCondition(
                    key=key, match=MatchText(text=value["$contains"])
                )
        else:
            return FieldCondition(key=key, match=MatchValue(value=value))
    raise ValueError(f"Cannot translate filter condition: {condition}")
