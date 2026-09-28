START
  ↓
retrieve
  ↓
grade_documents
  ↓
 ┌───────────────┐
 │               │
relevant       not relevant
 │               │
 ↓               ↓
generate       rewrite
 │               │
 ↓               │
grade_answer     │
 │               │
 ├── good ─────→ END
 │
 └── bad ──────→ retrieve
