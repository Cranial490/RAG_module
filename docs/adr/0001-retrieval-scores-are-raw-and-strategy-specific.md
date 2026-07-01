# Retrieval scores are raw and strategy-specific

Each retrieval strategy returns scores on its own scale (e.g. dense cosine in roughly 0.0–1.0, cross-encoder logits un-normalized, hybrid fusion outputs depending on the fusion function). The framework does **not** normalize scores at the strategy boundary, and does not pretend scores are comparable across strategies. Each strategy's docstring describes what its score means; callers interpret accordingly.

## Why

- Normalization to a common scale (e.g. squashing every score into `[0, 1]`) is information loss disguised as ergonomics. Once a cross-encoder's logits are squashed, relative magnitudes useful for thresholding or ensembling are gone. There is no canonical normalization function — the squash itself is a design decision per strategy.
- Cross-strategy score comparison is not a use case the framework has today. Adding the contract speculatively pays the cost (every new strategy must implement and justify a normalization) without clear benefit.
- Honesty over ergonomics: the framework's job is to expose what each strategy produced, not to hide what makes them different.

## Considered alternatives

- **Normalize to `[0, 1]` at the strategy boundary.** Rejected for the reasons above.
- **Raw scores plus a `score_kind` discriminator** (e.g. `"cosine"`, `"rerank_logit"`, `"rrf"`). Rejected for now as speculative; can be added additively later if a real consumer needs it.
