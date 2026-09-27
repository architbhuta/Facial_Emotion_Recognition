# Facial Emotion Recognition Capstone — Working Rules

One-night coursework capstone. See:
- `docs/superpowers/specs/2026-09-27-facial-emotion-recognition-design.md` — design/scope
- `docs/adr/0001-fer-capstone-stack.md` — key decisions and why
- `docs/superpowers/plans/2026-09-27-fer-capstone.md` — task-by-task implementation plan

## Ground rules

- **No novelty, no scope creep.** FER-2013 only, single CNN, Grad-CAM +
  saliency maps. Do not add CK+, hyperparameter search, ensembling, extra
  datasets, or a web/app frontend unless explicitly asked.
- **Test after every implementation step.** Never move to the next task
  without actually running the code and confirming the expected output —
  per the plan's per-task "Run and verify" steps. Don't mark a plan step
  done on the basis of it "should work."
- **Modest compute budget.** Target hardware is an 8GB M1 MacBook Air, no
  dedicated GPU. Keep batch sizes and epoch counts small; prefer a
  correct, fast-enough run over a longer one chasing accuracy.
- **~55–65% test accuracy is success, not a bug.** FER-2013 is a hard,
  imbalanced dataset for a simple CNN. Do not add complexity trying to
  push past this unless the user asks for it.
- **`fer_capstone.py` (jupytext) is the source of truth; `FER_Capstone.ipynb`
  is the generated deliverable.** Edit the `.py`, then regenerate/re-sync
  the notebook with `jupytext --to notebook fer_capstone.py` — don't hand-edit
  the `.ipynb` JSON directly.
- **Class index mapping is fixed:** alphabetical
  `["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]` →
  `0..6`. Don't silently reorder this anywhere in the pipeline.
- **Data files (`data/*.pt`, `checkpoints/`) are gitignored** — they're
  regenerable (see README.md) and shouldn't be committed.
- **Environment:** conda env `fer-capstone` (Python 3.11, see
  `environment.yml`) — the host's system Python (3.14) is intentionally not
  used because current stable PyTorch wheels don't support it yet.
- **Frequent, small commits** — one per plan task, not one giant commit at
  the end.
