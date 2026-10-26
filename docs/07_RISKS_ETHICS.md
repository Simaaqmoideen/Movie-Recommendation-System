# Risks and Ethics: Movie Recommendation System

| Risk | Mitigation |
|---|---|
| Sparse data gives weak collaborative results | Fall back to content-based rankings; blend community predictions only for mapped catalog titles. |
| Data leakage in evaluation | Split by time or hold out per-user ratings; never evaluate on training pairs. |
| Popularity bias | Compare against a popularity baseline; consider re-ranking. |

## Ethics and responsible-use notes
- Be honest about limitations in the README and report; do not overstate accuracy.
- Use only data you are allowed to use; anonymise any personal information.
- Document who could be harmed by a wrong output and how the design reduces that risk.
- New-user ratings are held in local Streamlit session state and are not persisted. If a TMDB key is configured, displayed title names are sent to TMDB for poster and overview lookup.
- Recommendation scores are relative ranking signals, not calibrated probabilities or endorsements.
