# Project 2: Movie Recommendation System

> Suggests movies based on what a user likes, the way streaming platforms do.

**What you will learn:** How platforms like Netflix recommend content: content-based filtering, collaborative filtering, similarity metrics, the cold-start problem and ranking evaluation.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py       # downloads the allowed MovieLens small dataset
python -m src.eda                  # data summary and EDA charts
python src/recommender.py          # content-based sample demo
streamlit run src/app.py
```
Run the tests with `python -m pytest -q` from this folder. The included sample works offline; MovieLens small is downloaded only for the collaborative model and evaluation. The EDA notebook is `notebooks/01_eda.ipynb`.

## Folder map
| Path | Purpose |
|---|---|
| `docs/01_PROJECT_BRIEF.md` | Problem, users, scope, success criteria |
| `docs/02_REQUIREMENTS.md` | Must / should / could requirements and non-functional needs |
| `docs/03_ARCHITECTURE.md` | Design, components, tech stack |
| `docs/04_DATA.md` | Data sources, schema and data rules |
| `docs/05_ROADMAP.md` | Phase-by-phase build plan with checklists |
| `docs/06_TESTING_EVALUATION.md` | Test plan and evaluation metrics |
| `docs/07_RISKS_ETHICS.md` | Risks, mitigations, ethics notes |
| `docs/08_DELIVERABLES_CHECKLIST.md` | Definition of done |
| `docs/09_REPORT_TEMPLATE.md` | Final report / presentation outline |
| `src/` | Reusable recommender, data, and evaluation logic |
| `tests/` | Automated tests |
| `data/` | Datasets (git-ignored except samples) |
| `notebooks/01_eda.ipynb` | MovieLens data exploration |
| `reports/` | EDA charts and evaluation metrics |

## Tech stack
- Python 3.10+
- pandas, numpy
- scikit-learn (TF-IDF, cosine similarity)
- scikit-surprise or implicit (optional, for collaborative filtering)
- Streamlit (UI)
- pytest
