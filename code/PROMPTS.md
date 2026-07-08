# Sample prompts for Lesson 6 — RAG

These prompts let you explore and modify this lab using a coding agent. Open the
lab's `code/` folder as the agent's working directory. Run
`python 01-simple-rag/ingest.py` once first — most prompts assume the index exists.

## Explore — understand what each exercise does

- **"Walk me through the sliding-window loop in `chunk_document()` in `01-simple-rag/ingest.py`. What happens to a sentence that straddles a chunk boundary, and how does `OVERLAP_WORDS` rescue it?"**
  The agent should trace the window arithmetic and show why boundary text appears in two chunks — the reason overlap exists at all.

- **"In `01-simple-rag/ingest.py`, `embed_text()` uses `invoke_model` with a JSON body, while `rag_answer.py` uses `converse`. Why two different Bedrock APIs in one lab?"**
  Takeaway: `converse` is the chat abstraction; embedding models have no chat shape, so they're called through the raw per-model API — and each embedding model has its own body format.

- **"Run `python 01-simple-rag/search.py 'can I work a second job?'` and explain why the moonlighting chunks match even though they never contain the words 'second job'."**
  The core semantic-search takeaway: embeddings match meaning, not keywords — this exact query is where vector search beats grep.

- **"Open `data/faiss_index_handbook/metadata.json` and explain why the chunk texts and the embedding model id live there instead of inside `index.faiss`. What would break if I rebuilt the index with a different embedding model but kept querying with the old one?"**
  FAISS stores only vectors — texts ride separately, and results are just row numbers. Model mismatch fails *silently* with garbage neighbors, which is why the model id is recorded.

- **"Compare `answer()` in `01-simple-rag/rag_answer.py` with `run_agent()` in `02-rag-as-tool/rag_tool.py`. Both end up with the same chunks in the prompt — trace how each gets there, and who decides that retrieval happens."**
  The pipeline-vs-agentic contrast: hardwired retrieval on every input vs. a model-issued `search_handbook` call — and in the tool version the *model* writes the search query.

- **"In `01-simple-rag/rag_answer.py`, what work does the SYSTEM prompt do? Predict what changes if I delete the 'ONLY' rules, then check your prediction against the Modify prompt below."**
  Grounding lives in the prompt, not the pipeline — retrieval alone doesn't stop the model from using its training data.

- **"Run `python 02-rag-as-tool/rag_by_role.py employee` and then `manager`, ask both 'How should I run 1:1s with my report?', and explain the two different behaviors. Where exactly is access control enforced — what stops the employee agent from ever reading the playbook?"**
  Access lives in `tools.plan_tool`: the `plan_sources` tool's `source` enum is restricted to `ROLE_TOOLS[role]`, so the employee's planner can't even name `search_manager_playbook` — there's nothing to jailbreak.

- **"`rag_by_role.py` uses a forced `plan_sources` call, but `rag_tool.py` just lets the model call the search tool in a loop. Read both and explain why rag_by_role plans first. Then ask `rag_by_role.py manager` a two-part question like 'what's our severance policy in weeks, and how do I run the termination conversation?' and watch it route to both corpora."**
  Nova 2 Lite reliably emits only ONE tool call per turn, so a two-corpus question in the agent loop searches just one; forcing a single structured plan makes the model enumerate every source it needs — the plan-then-act pattern that fixes multi-source retrieval.

- **"Read `map_kbs()` and the `for kb in kbs` loop in `02-rag-as-tool/ingest_kbs.py`, then trace what one iteration produces. Why does every audience end up as a totally separate index, and what can a single `search_kb` call NOT do that one combined index could?"**
  The scaling pain of per-audience indexes — separate stores to maintain, no cross-corpus ranking — the exact problem Lesson 7's metadata filtering exists to solve.

## Modify — change the code to see a concept in action

- **"In `01-simple-rag/ingest.py`, set `CHUNK_WORDS = 60` and `OVERLAP_WORDS = 10`, re-run ingest, then ask `rag_answer.py` 'what benefits do I get?' before and after. Compare the chunk count, the retrieved chunk ids, and the answer quality."**
  Chunk size is the biggest naive-RAG knob: tiny chunks retrieve precisely but starve the model of context. (Restore 250/50 and re-ingest afterwards.)

- **"Change `TOP_K` in `01-simple-rag/search.py` from 4 to 1, then ask `rag_answer.py` a question that spans topics, like 'compare the vacation policy with the sabbatical policy'. What did the single chunk miss?"**
  Top-k trades context breadth against noise — multi-topic questions are exactly where k=1 breaks.

- **"In `01-simple-rag/rag_answer.py`, delete the grounding rules from `SYSTEM` (keep just 'You are the NovaOps employee assistant.') and ask: 'What is NovaOps' policy on unlimited PTO?' Compare with the grounded version."**
  The ungrounded model invents a plausible policy from training data; the grounded one says the context doesn't cover it. This is the RAG failure students will meet in production.

- **"Make `rag_answer.py` print the full prompt it sends to Nova, plus the `usage` token counts from the response. How many input tokens does one top-4 question cost, and where do they come from?"**
  Makes "stuffing context into the prompt" literal — retrieved chunks are just paid input tokens, which is why top-k and chunk size are cost knobs too.

- **"In `02-rag-as-tool/rag_tool.py`, ask 'summarize the moonlighting policy AND the device policy for leavers'. Does the model call `search_handbook` once or twice? Print each query it writes and compare them to my question."**
  Agentic RAG can decompose a question into multiple targeted searches — something the fixed pipeline in `rag_answer.py` structurally cannot do.

- **"Swap the retrieval in `01-simple-rag/search.py` to plain keyword scoring (count query words in each chunk) behind the same `search()` signature, and compare its results with the vector version on 'can I work a second job?'."**
  Builds the intuition for *why* embeddings — and previews Lesson 7's hybrid search, where the two approaches get combined.

- **"Merge both corpora into a single index: copy `01-simple-rag/ingest.py` so it indexes `data/handbook` AND `data/manager_playbook` together, then point `rag_by_role.py` at one search tool over the merged index, and ask an employee-role question about performance reviews. What leaked, and what would each vector need to carry for the employee/manager split to survive a shared index?"**
  Deliberately breaks the naive separation to arrive at the answer yourself: per-chunk metadata plus filtered search — which is exactly where Lesson 7 starts.
