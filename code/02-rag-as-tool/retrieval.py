"""Vector retrieval layer for the tool-based RAG demos.

The same embed -> load -> search machinery as 01-simple-rag/search.py, but
parameterized by index directory so one set of functions serves any corpus.
rag_tool.py and rag_by_role.py import from here — neither re-implements retrieval.
"""
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

CODE_DIR = Path(__file__).resolve().parents[1]
TOP_K = 4

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def embed_text(text: str, model_id: str) -> list[float]:
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def load_kb(index_dir: Path, build_cmd: str) -> dict:
    """Load one corpus's index + chunk texts + embedding-model id into memory
    (see 01-simple-rag/search.py). Returns a 'kb' bundle that search() reads."""
    if not (index_dir / "index.faiss").exists():
        raise SystemExit(f"No index at data/{index_dir.name}. Build it first:  {build_cmd}")
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "index": faiss.read_index(str(index_dir / "index.faiss")),
        "chunks": metadata["chunks"],
        "model_id": metadata["embedding_model"],
    }


def search(kb: dict, query: str, top_k: int = TOP_K) -> list[tuple[int, float, str]]:
    """Top-k cosine search over one loaded KB. Returns (chunk_id, cosine
    similarity, text), best first. Callers format it into prompt context and/or
    print the scores (tools.py adds the source tag)."""
    query_vector = np.array([embed_text(query, kb["model_id"])], dtype="float32")
    scores, ids = kb["index"].search(query_vector, top_k)  # cosine similarity
    return [(int(i), float(s), kb["chunks"][int(i)]) for i, s in zip(ids[0], scores[0])]
