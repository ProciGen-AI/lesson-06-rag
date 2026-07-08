"""Full RAG loop: retrieve handbook chunks, stuff them into the prompt, generate.

Interactive Q&A over the handbook:

    python 01-simple-rag/rag_answer.py
"""
import os

import boto3
from dotenv import find_dotenv, load_dotenv

# search.py is the whole retrieval layer; this file only adds generation.
from search import TOP_K, load_index_and_chunks, search

load_dotenv(find_dotenv())

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]

# The grounding rules are what makes this RAG rather than a regular chat:
# without them the model happily answers from its training data and the
# retrieved context changes nothing.
SYSTEM = (
    "You are the NovaOps employee assistant. Answer questions using ONLY the "
    "context chunks provided in the user message. If the context does not "
    "contain the answer, say so plainly — do not fill gaps with outside "
    "knowledge or guesses."
)


def format_context(results: list[tuple[int, float, str]]) -> str:
    return "\n\n".join(f"[chunk {chunk_id}]\n{text}" for chunk_id, _distance, text in results)


def answer(question: str, results: list[tuple[int, float, str]]) -> str:
    # "Stuffing": the retrieved chunks are pasted into the user message, and
    # that's the only channel through which the model sees the handbook.
    prompt = (
        f"Context:\n{format_context(results)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"]


def main() -> None:
    index, chunks, embedding_model_id = load_index_and_chunks()
    print(f"RAG over {index.ntotal} handbook chunks (top-{TOP_K} retrieval).")
    print("Ask a question — empty line or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break

        results = search(question, index, chunks, embedding_model_id)
        retrieved = ", ".join(f"{chunk_id} (cos {similarity:.3f})" for chunk_id, similarity, _ in results)
        print(f"  retrieved chunks: {retrieved}")
        print(f"\n{answer(question, results)}\n")


if __name__ == "__main__":
    main()
