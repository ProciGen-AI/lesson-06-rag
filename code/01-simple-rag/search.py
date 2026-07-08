"""Retrieval client: embed a question, return the top-k closest handbook chunks.

"Closest" means highest COSINE SIMILARITY between the question's embedding and
each chunk's embedding — that similarity score is what ranks retrieval.

Run it standalone to see exactly what retrieval returns before any LLM is
involved:

    python 01-simple-rag/search.py "How many weeks of paid vacation do I get?"

rag_answer.py imports load_index_and_chunks() and search() from here — this
file is the whole retrieval layer.
"""
import json
import os
import sys
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

CODE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = CODE_DIR / "data" / "faiss_index_handbook"
TOP_K = 4

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def embed_text(text: str, model_id: str) -> list[float]:
    # Same call as ingest.py. The query MUST be embedded with the same model
    # (and dimensions) the index was built with — vectors from different models
    # live in unrelated spaces, so a mismatch doesn't error, it just returns
    # nonsense neighbors. That's why ingest.py records the model in metadata.json
    # and load_index() hands it back here.
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def load_index_and_chunks() -> tuple[faiss.Index, list[str], str]:
    """Load into memory what ingest.py saved to disk: the FAISS vector index,
    the chunk texts (FAISS stores only vectors, so the texts ride alongside in
    metadata.json), and the embedding-model id the query must be embedded with.
    Called once at startup; the returned values are reused for every search."""
    if not (INDEX_DIR / "index.faiss").exists():
        raise SystemExit("No index found. Build it first:  python 01-simple-rag/ingest.py")
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    chunks = metadata["chunks"]
    # Returned so search() can embed the query with the SAME model the index was
    # built with — a mismatch retrieves nonsense silently, it never errors.
    embedding_model = metadata["embedding_model"]
    return index, chunks, embedding_model


def search(
    question: str,
    index: faiss.Index,
    chunks: list[str],
    model_id: str,
    top_k: int = TOP_K,
) -> list[tuple[int, float, str]]:
    """Return the top_k chunks as (chunk_id, cosine similarity, text), best first."""
    # The query has to be embedded too — with the SAME model the documents were,
    # so it lands in the same vector space (that's what model_id is for). Only
    # then can we compare it against the stored chunk vectors.
    query_vector = embed_text(question, model_id)
    # FAISS searches a batch of queries at once, so wrap the one vector in a 2D
    # float32 array of shape (1, dim).
    query = np.array([query_vector], dtype="float32")
    # This is the heart of retrieval: COSINE SIMILARITY. The query and every
    # chunk are unit vectors, so their inner product is their cosine similarity —
    # 1.0 = same direction (same meaning), 0.0 = unrelated. The IndexFlatIP index
    # returns that score and gives back the top_k highest-scoring chunks first.
    cosine_similarities, ids = index.search(query, top_k)  # both come back shaped (1, top_k)
    return [(int(i), float(score), chunks[int(i)]) for i, score in zip(ids[0], cosine_similarities[0])]


def print_results(question: str, results: list[tuple[int, float, str]]) -> None:
    print(f"Question: {question}")
    for chunk_id, similarity, text in results:
        preview = text[:400] + ("..." if len(text) > 400 else "")
        print(f"\n--- chunk {chunk_id}  (cosine similarity {similarity:.4f} — higher is closer) ---")
        print(preview)


def main() -> None:
    # Retrieval on its own, no LLM — this is what rag_answer.py feeds to the
    # model. Runs a sample question; pass your own as an arg to try others.
    question = " ".join(sys.argv[1:]) or "How many weeks of paid vacation do I get?"
    index, chunks, model_id = load_index_and_chunks()
    results = search(question, index, chunks, model_id)
    print_results(question, results)


if __name__ == "__main__":
    main()
