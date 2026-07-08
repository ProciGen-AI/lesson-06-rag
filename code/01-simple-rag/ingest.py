"""Ingest the NovaOps employee handbook: load -> chunk -> embed -> store.

Builds the FAISS index that search.py and rag_answer.py query. Run it once
before the other scripts (and re-run after changing the corpus or chunking):

    python 01-simple-rag/ingest.py
"""
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Paths are anchored to this file, not the shell's cwd, so the script runs
# correctly from code/ or from inside this folder.
CODE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = CODE_DIR / "data" / "handbook"
# Named for its corpus (faiss_index_<name>) — the same index the 02-rag-as-tool
# demos reuse, so building it here or there produces one shared handbook index.
INDEX_DIR = CODE_DIR / "data" / "faiss_index_handbook"

CHUNK_WORDS = 250   # max words per chunk
OVERLAP_WORDS = 50  # words shared between consecutive chunks

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
EMBEDDING_MODEL_ID = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]


def load_documents() -> list[str]:
    return [path.read_text(encoding="utf-8") for path in sorted(DOCS_DIR.glob("*.md"))]


def chunk_document(text: str) -> list[str]:
    # Naive fixed-size chunking: slide a CHUNK_WORDS window, stepping
    # CHUNK_WORDS - OVERLAP_WORDS at a time. The overlap keeps a passage that
    # straddles a boundary retrievable from at least one chunk. Smarter,
    # structure-aware chunking is Lesson 7 material.
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def chunk_documents(documents: list[str]) -> list[str]:
    # Chunk each document separately — a chunk must never span two files, or
    # retrieval could return text that mixes two unrelated policies.
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    print(f"Chunked {len(documents)} documents -> {len(chunks)} chunks")
    return chunks


def embed_text(text: str) -> list[float]:
    # Embedding models are called via invoke_model with a model-specific JSON
    # body — converse() is chat-only. Titan embeds ONE text per call; there is
    # no batch API, hence the loop in main(). normalize=True returns unit-length
    # vectors, so cosine similarity between two chunks is just their inner
    # product — which is exactly what the index searches by (see save_index).
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    # The slow (and only paid) part of ingest — one Bedrock call per chunk,
    # since Titan has no batch API. Hence the progress prints.
    embeddings = []
    for i, chunk in enumerate(chunks, start=1):
        embeddings.append(embed_text(chunk))
        if i % 20 == 0 or i == len(chunks):
            print(f"  embedded {i}/{len(chunks)}")
    # Stack into the (num_chunks, 1024) float32 matrix FAISS expects.
    return np.array(embeddings, dtype="float32")


def save_index(vectors: np.ndarray, chunks: list[str]) -> None:
    # IndexFlatIP searches by inner product. Our vectors are unit-normalized, so
    # the inner product IS cosine similarity — this is a cosine-similarity index.
    # "Flat" = each query is compared against every vector (exact, brute force);
    # approximate indexes only start to matter at millions of vectors.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    # FAISS stores only the vectors. The chunk TEXTS must be kept separately —
    # a search result is just a row number into this list. The embedding model
    # id is recorded too: queries must be embedded with the same model, or
    # search silently returns garbage neighbors instead of erroring.
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL_ID, "chunks": chunks}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved FAISS index ({index.ntotal} vectors, dim {vectors.shape[1]}) to {INDEX_DIR}")


def main() -> None:
    # The whole naive-RAG ingest, one stage per line.
    documents = load_documents()
    chunks = chunk_documents(documents)
    vectors = embed_chunks(chunks)
    save_index(vectors, chunks)


if __name__ == "__main__":
    main()
