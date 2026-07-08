# Exercise 00 — AWS setup for this lab

If your `.env` already works for Lessons 2–5, there is exactly **one** new thing
to set up here: access to the **embedding model**. This lab embeds documents and
queries with **Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`), which
needs its own model-access grant — separate from Nova's.

(New to the course, or no AWS setup yet? Do Lesson 2's `00-aws-setup.md` first —
account, IAM user, access keys — then come back here.)

## Step 1 — Enable Titan Text Embeddings V2

1. Open the Bedrock console (region `us-east-1` or whatever your `AWS_REGION` is).
2. **Discover → Model catalog → Model access** (or the **Model access** page directly).
3. Find **Titan Text Embeddings V2** (provider: Amazon) and enable it.
   Access is usually granted instantly for Amazon models.

Two things worth knowing:

- Unlike Nova, the embedding model is called with its **bare ID** —
  `amazon.titan-embed-text-v2:0`, no `us.` inference-profile prefix. Embedding
  models are invoked on-demand in-region.
- If your IAM policy allowlists **specific model ARNs** instead of granting
  Bedrock-wide access, add
  `arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0` to it —
  otherwise you'll get an `AccessDeniedException` even with model access enabled.

## Step 2 — Verify with `setup.sh`

> **On Windows?** Run from **Git Bash** (easiest via VS Code's integrated
> terminal set to Git Bash) — PowerShell/cmd can't `source` a `.sh`. No Git
> Bash? Install [Git for Windows](https://git-scm.com/download/win) or
> [WSL](https://learn.microsoft.com/windows/wsl/install). Started in
> PowerShell? Run `.\setup.ps1` and it points you to Git Bash.

From the lab's `code/` folder:

```bash
source setup.sh
```

This lab's setup script checks **both** models at the end:

```
✓ Bedrock OK — model replied: 'ok'
✓ Embeddings OK — amazon.titan-embed-text-v2:0 returned a 256-dim vector
```

Two green checks = you're ready for `01-simple-rag/ingest.py`. If the embedding
check fails, the error message tells you which of the two causes above to fix.

(The smoke test asks Titan for a small **256**-dim vector just to prove access;
the lab itself embeds at **1024** dimensions — see `dimensions` in
`01-simple-rag/ingest.py`. Titan supports 256/512/1024; more dimensions carry
more detail at a larger index.)

## What this costs

Titan Text Embeddings V2 is priced per input token and is one of the cheapest
models on Bedrock. Indexing the whole handbook corpus (~80 chunks) costs a
fraction of a cent, and each question embeds only the query — so re-running
`ingest.py` freely is fine.
