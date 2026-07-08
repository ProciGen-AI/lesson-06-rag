# Lesson 6 — RAG (retrieval-augmented generation)

The model has never read your company's documents, and you can't paste a whole
handbook into every prompt. **RAG** is the standard fix: index the documents
once, then for each question *retrieve* the few most relevant excerpts and stuff
them into the prompt. This lab builds that pipeline from scratch — no framework,
every moving part in plain Python — over the employee handbook of **NovaOps**,
the fictional company whose internal assistant you'll build as the course's
final project.

## The two parts

**Part 1 — [`01-simple-rag/`](code/01-simple-rag/): the naive end-to-end pipeline.**
[`ingest.py`](code/01-simple-rag/ingest.py) → [`search.py`](code/01-simple-rag/search.py) → [`rag_answer.py`](code/01-simple-rag/rag_answer.py),
one file per stage: **load → chunk → embed → store (FAISS) → retrieve top-k →
stuff into the prompt → generate**. `ingest` builds the index, `search` is the
retrieval client, and `rag_answer` adds grounded generation on top of it.

**Part 2 — [`02-rag-as-tool/`](code/02-rag-as-tool/): retrieval as a tool, then role-based access.**
[`rag_tool.py`](code/02-rag-as-tool/rag_tool.py) wraps retrieval as a **tool** in
Lesson 4's agent loop — the model *decides* when to search.
[`rag_by_role.py`](code/02-rag-as-tool/rag_by_role.py) then adds a second corpus
(the **manager playbook**) and role-based access: a **plan → retrieve → answer**
flow where a forced planning step routes each question to the corpora it needs,
restricted to the role's allowed sources. Both run over shared helpers —
[`retrieval.py`](code/02-rag-as-tool/retrieval.py) is the vector layer, and
[`tools.py`](code/02-rag-as-tool/tools.py) holds the tool schemas, the plan tool,
the search operation, and the role→tools map — with the indexes built by
[`ingest_kbs.py`](code/02-rag-as-tool/ingest_kbs.py).

The contrast between the parts is the lesson's punchline: `rag_answer.py`
retrieves on *every* input (ask it "hello" and watch it stuff handbook chunks
anyway), while `rag_tool.py` retrieves only when the model thinks it needs to.

`rag_by_role.py` adds a second punchline that sets up Lesson 7. Managers may
read both the handbook and the manager playbook; employees only the handbook —
and the access control is **which corpora a role's plan step may route to** (the
`plan_sources` tool's `source` enum is restricted to `ROLE_TOOLS[role]`, so the
model can't even name a corpus its role can't read). Run it with no argument for
a scripted demo: the same 1:1s question as `employee` vs `manager`, then two
manager questions that each pull from **both** corpora. (Why a planning step
instead of `rag_tool.py`'s loop? Nova 2 Lite reliably makes only one tool call
per turn, so a two-corpus question would search just one; forcing the model to
*plan* the sources it needs makes multi-source retrieval reliable.)

The `for kb in kbs` loop in `ingest_kbs.py` is the whole cost of this design:
every audience is a wholly separate index to build, store, and re-embed, and no
single search ranks across two corpora. Every new audience is another index.
Lesson 7 solves the same problem with **one** index and metadata filtering.

## Learning objectives

- What each pipeline stage is for, and what actually lives in a vector index
  (vectors in FAISS; the chunk **texts** ride separately in `metadata.json`).
- Embeddings as the retrieval engine: same-model-for-index-and-query, why
  "can I work a second job?" finds the moonlighting policy without sharing a
  single keyword with it.
- Naive chunking (fixed window + overlap) and the knobs it exposes
  (`CHUNK_WORDS`, `OVERLAP_WORDS`, `TOP_K`).
- Grounding: the system-prompt rules that force the model to answer from
  retrieved context instead of its training data.
- Pipeline RAG vs. tool-based (agentic) RAG — when retrieval is hardwired vs.
  model-decided.
- Reading a similarity score: cosine values are **model-specific** (Titan's
  "strong match" is ~0.4–0.5, not 0.8) — what matters is a chunk's score
  *relative to the others*, not an absolute threshold.
- Query planning does two jobs at once: **routing** a question to the corpora it
  needs, and **rewriting** a focused query per corpus (which beats searching a
  source with a raw multi-topic question — see `rag_by_role.py`'s output).
- Naive multi-audience access control: separation by tool registration over
  per-audience indexes — and why it stops scaling past two audiences.

Deliberately **out of scope** (Lesson 7): reranking, hybrid/keyword search,
query rewriting, metadata filtering, smarter chunking, and RAG evaluation.

## Prerequisites

- **Lesson 2** — Bedrock Converse basics. **Lesson 4** — the tool loop
  (part 02 reuses it).
- AWS setup from Lessons 2–5 **plus one new model grant**: Titan Text
  Embeddings V2 — see [`code/00-aws-setup.md`](code/00-aws-setup.md).

## How to run

> **On Windows?** Run from **Git Bash** (easiest via VS Code's integrated
> terminal set to Git Bash) — PowerShell/cmd can't `source` a `.sh`. No Git
> Bash? Install [Git for Windows](https://git-scm.com/download/win) or
> [WSL](https://learn.microsoft.com/windows/wsl/install). Started in
> PowerShell? Run `.\setup.ps1` and it points you to Git Bash. macOS/Linux:
> commands work as-is.

```bash
git clone https://github.com/ProciGen-AI/lesson-06-rag.git
cd lesson-06-rag/code
source setup.sh
```

The setup ends with **two** smoke tests — Nova (generation) and Titan
(embeddings). Then run the pipeline in order, from `code/`:

```bash
python 01-simple-rag/ingest.py                      # build the index (once)
python 01-simple-rag/search.py "how much vacation do I get?"   # retrieval only
python 01-simple-rag/rag_answer.py                  # interactive RAG Q&A
python 02-rag-as-tool/ingest_kbs.py                 # one index per corpus under data/ (once)
python 02-rag-as-tool/rag_tool.py                   # the agent decides WHEN to search
python 02-rag-as-tool/rag_by_role.py                # scripted demo: employee vs manager, cross-source
python 02-rag-as-tool/rag_by_role.py employee       # interactive: handbook only
python 02-rag-as-tool/rag_by_role.py manager        # interactive: handbook + playbook
```

Good questions to try: vacation and benefits, the moonlighting policy, what
happens to your laptop when you leave, how promotions work. Then ask something
the handbook can't answer ("what's our AWS bill?") and watch the grounding
rules kick in.

**Reading `rag_by_role.py`'s output.** For each question it prints the *plan*
(which corpora, and the query it wrote for each), the chunks + cosine scores it
retrieved with those planned queries (**used** in the answer), and — for
comparison only, **not** sent to the model — the chunks the *raw* question would
have retrieved from the same corpora. Watch the two cross-source manager
questions: the tailored query scores noticeably higher on the corpus that the
question's *other* half was diluting (the playbook for severance, the handbook
for parental leave). That gap is the payoff of planning a query per source.
Scores wobble run to run (the model rewrites the queries fresh each time), but
the dilution gap is stable.

## The corpus

`code/data/handbook/` holds NovaOps' employee handbook — 15 markdown files
(benefits, how-we-work, titles and levels, device policy, …). It's the same
world and document set the final project's assistant answers questions over;
the index you build here is the project's first retrieval layer.
`code/data/manager_playbook/` is the second, **managers-only** corpus — 17
files on 1:1s, feedback, hiring, and performance management — used by
`rag_by_role.py`. (Both are adapted from Basecamp's public employee handbook and
manager playbook.)

## Build it yourself

Once you've studied the lab, rebuild its tool-based RAG assistant — the one where
the model *decides when to search* ([`rag_tool.py`](code/02-rag-as-tool/rag_tool.py))
— from a spec: see [`SDD/`](SDD/). You drive a coding agent from `SDD/PROMPT.md`
in a clean repo (raw handbook, no prebuilt index — building it is part of the
task), then score the result with the `validate-lab` skill.
