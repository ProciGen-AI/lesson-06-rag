# Lesson 6 — RAG · Spec-Driven Build (SDD)

The build-it-yourself half of Lesson 6. You've studied the worked lab in
[`../code/`](../code/); now **rebuild** its tool-based RAG assistant — the one
where the model *decides when to search* ([`../code/02-rag-as-tool/rag_tool.py`](../code/02-rag-as-tool/rag_tool.py))
— from a spec, driving a coding agent with `PROMPT.md`, then score yourself with
the `validate-lab` skill.

## How it works

You build in a **separate, clean repo** so your agent starts from a blank slate —
no answer key, no rubric to game, no dependency hints. That repo ships the raw
handbook under `data/handbook/` but **no prebuilt index** — building the index is
part of the exercise. This `SDD/` folder (inside the lesson repo) holds the
**task** (`PROMPT.md`) and the **validator** (`validate-lab/`); the building
happens in the other repo.

## 1. Get a clean build workspace

```bash
git clone https://github.com/ProciGen-AI/lesson-06-rag-sdd.git
cd lesson-06-rag-sdd
# copy or create a working .env in this folder (your AWS creds + BEDROCK_MODEL_ID
# + BEDROCK_EMBEDDING_MODEL_ID) before the next line
source setup.sh                  # venv + base deps (boto3, python-dotenv) + Bedrock + embedding smoke test
mv CLAUDE.md-example CLAUDE.md   # activate the build conventions your agent reads
```

> The build repo is deliberately bare — **setup + data + conventions only**, no
> `PROMPT.md`, no `requirements.txt`, no answer. It has no `.env.example` either:
> **bring your own `.env`** — copy the one you already filled in for this lesson
> (e.g. `cp ../lesson-06-rag/.env .env`, adjust the path to wherever you cloned
> the lesson repo). This lab needs **two** models enabled: Nova (generation) and
> Titan Text Embeddings V2 (`BEDROCK_EMBEDDING_MODEL_ID`) — `setup.sh` smoke-tests
> both.
>
> **On Windows?** Run these from Git Bash (easiest via VS Code's integrated terminal).

## 2. Build

Hand your agent the task — **`PROMPT.md` in this `SDD/` folder** (open it and paste
it, or point your agent at it). It's a *partial* spec; the load-bearing decisions
are yours (the `▢ YOU DECIDE` block — chunking, embeddings, the search tool's
schema and description, the agent loop, the grounding rules). Build `rag_tool.py`
in the build repo, build the index once, run it, iterate:

```bash
python ingest.py        # (or whatever you named it) — build the index once
python rag_tool.py      # try a greeting first, then a policy question
```

## 3. Validate

Back here in the lesson repo's `SDD/` folder, invoke the **validate-lab** skill. It
asks for the path to your build repo, then scores your `rag_tool.py` against the
reference `../code/02-rag-as-tool/rag_tool.py` out of 100 — naming what's missing /
weaker / better, with a cheat prompt if you're out of time.
