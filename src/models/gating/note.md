Your current approach: feature-driven gating (FastText LID → language-specific TF-IDF + keyword voting → fallback rules) packaged in a pluggable gating interface. This is deterministic, explainable, and low-latency.

A strong orthogonal alternative should differ in both representation and decision policy while remaining lightweight. Two solid candidates:

### Embedding-based nearest-centroid router (language-agnostic)

- Replace TF-IDF/keywords with multilingual sentence embeddings (e.g., LaBSE, MiniLM-multilingual, or a small sentence-transformer).

- Build per-domain, per-language centroids from a few labeled examples.

- At inference: detect language, embed prompt, compute cosine similarity to centroids, pick top domain; confidence from margin between top-1 and top-2.

- Pros: language-agnostic semantics; robust to synonyms and translation variation; very small training need.

- Cons: slightly higher latency than TF-IDF; embedding model is a dependency.

### Translate-then-classify with robust English-only classifier

Detect language → translate to English → classify with a single strong but compact English model (e.g., TF-IDF+LinearSVM or small transformer).

Add uncertainty gating based on translation quality heuristics (length ratio, OOV rate, numeric/token preservation).

Pros: only one classifier to train/maintain; leverages rich English data.

Cons: error propagation from MT; worse on code-switching; added latency.

If you want one orthogonal option, choose (1) nearest-centroid router. It’s simple, explainable, and differs fundamentally from sparse lexical features.

How to integrate the orthogonal router cleanly
Add another gating strategy implementing the same BaseGating interface:

### EmbeddingGating

init(model, domain_centroids) where domain_centroids is {lang: {‘edu’: vector, ‘fin’: vector}}

decide(text, lang, …) → embed(text) → cosine to centroids[lang] (fallback to ‘xx’ global centroids if missing) → return domain, confidence=margins.

This lets you A/B test:

HeuristicGating vs EmbeddingGating

Or even ensemble both with a meta-gate: if agreement → accept; if disagreement → pick higher-confidence; if both low → fallback rules.

Minimal plan to stand up the orthogonal router
Collect 50–200 short prompts per domain per language (or translate English seeds).

Compute sentence embeddings; average per domain to get centroids.

Store centroids to disk; load at runtime.

Use cosine similarity; margin threshold (e.g., 0.1) to trigger fallback.

Evaluation checklist (to choose the winner)
Accuracy/F1 by language and by domain.

Latency p50/p95 end-to-end.

Confidence calibration (does low confidence correlate with errors?).

Robustness: code-switching, numeric density, OOV jargon.

Drift sensitivity: re-evaluate monthly with new traffic.