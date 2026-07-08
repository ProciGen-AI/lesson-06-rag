<!-- ──────────────────────────────────────────────────────────────────────────
  HOW TO USE THIS FILE  (guidance to you, the student — NOT part of the prompt)

  Everything inside these <!-- ... --> comments is for you; your editor greys
  them out, so the plain text left over is the prompt itself.

  1. Rename CLAUDE.md-example -> CLAUDE.md so your agent reads the conventions.
  2. Hand the plain text below to your coding agent. It's a *partial* prompt —
     the decisions at the bottom are left out on purpose. That's the exercise.
  3. When you think it's done: build your index, run `python rag_tool.py`, try a
     greeting and then a policy question, then run the validate-lab skill.
─────────────────────────────────────────────────────────────────────────── -->

# Build: rag_tool.py

Build an assistant over the NovaOps employee handbook (`data/handbook/`).

Write it as `rag_tool.py` at the root of your build repo.

<!-- The repo ships the raw handbook but no prebuilt index — building it is part of the job. -->

<!-- ▢ YOU DECIDE — left out on purpose; this is the exercise:
       - how you turn the handbook into something searchable (and what a query
         "matching" a chunk even means);
       - how retrieval becomes a tool the model chooses to call, and the loop
         around that choice;
       - the rules that keep the answer to what was actually retrieved. -->
