# Further reading — Lesson 6 RAG

- **The original RAG paper** — Lewis et al., 2020, ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401). Where the term comes from; the modern "stuff chunks into the prompt" pattern is a simplification of this.
- **FAISS wiki** — [github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki). What lives beyond `IndexFlatIP`: approximate indexes (IVF, HNSW) for when brute force stops scaling.
- **Titan Text Embeddings V2** — [AWS docs](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html). The request/response shape used by `embed_text()`, plus the 256/512/1024 dimension trade-off.
- **ChromaDB docs** — [docs.trychroma.com](https://docs.trychroma.com/). The vector *database* counterpart to the FAISS *library* — the optional upgrade if you swap the lab's in-memory index for a persistent store (and where Lesson 7 heads next).
- **Anthropic's "Contextual Retrieval" post** — [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval). A readable tour of why naive RAG misses things — good preview of Lesson 7 (hybrid search, reranking).
