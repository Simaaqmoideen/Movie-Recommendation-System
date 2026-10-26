# Risks and Ethics: Movie Recommendation System

| Risk | Mitigation |
|---|---|
| Sparse data gives weak collaborative results | Fall back to content-based; use hybrid. |
| Data leakage in evaluation | Split by time or hold out per-user ratings; never evaluate on training pairs. |
| Popularity bias | Compare against a popularity baseline; consider re-ranking. |

## Ethics and responsible-use notes
- Be honest about limitations in the README and report; do not overstate accuracy.
- Use only data you are allowed to use; anonymise any personal information.
- Document who could be harmed by a wrong output and how the design reduces that risk.
