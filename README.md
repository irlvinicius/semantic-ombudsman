# semantic-ombudsman

Semantic triage prototype for citizen complaints sent to a municipal ombudsman office, built with sentence embeddings, vector search and text chunking. Academic project for an applied NLP course.

## The problem

The ombudsman office receives about 4,000 free-text complaints per month (potholes, broken street lights, slow healthcare service...) and triages them with keyword search. That causes three issues:

- **Undetected duplicates:** "asfalto esburacado" and "rua com buraco" are treated as different complaints.
- **Related topics are not grouped:** "falta de remédio no posto" and "demora na consulta" should belong to the same public health cluster.
- **Long complaints lose context** when indexed as a single block.

This project tackles them with dense vector representations.

## What is inside

| Deliverable | File | Description |
|---|---|---|
| 1 | `notebooks/01_analise_comparativa.ipynb` | BoW vs TF-IDF vs embeddings on three pairs of complaints |
| 2 | `notebooks/02_deteccao_duplicatas.ipynb` | Similarity matrix, duplicate detection and false positive/negative analysis |
| 3 | `notebooks/03_chunking_manifestacoes.ipynb` | Chunking of long complaints with different `chunk_size` / `chunk_overlap` |
| 4 | `app_ouvidoria.py` | Streamlit app: semantic search, full base, vector space and chunking |

## Main findings

- Sparse representations (BoW, TF-IDF) do not reach high similarity even between complaints about the same problem, because they depend on shared vocabulary.
- With `paraphrase-multilingual-MiniLM-L12-v2`, the default threshold of 0.85 finds no duplicate pair in this dataset. The threshold was calibrated on the data (0.68), favoring recall: 8 of 9 manually labeled duplicate pairs are found, at the cost of low precision.
- Long complaints follow a similar template ("the community requests an inspection..."), so their chunks cluster by role in the text more than by topic.

## Tech stack

Python 3.11 · [uv](https://docs.astral.sh/uv/) · scikit-learn · sentence-transformers · LangChain text splitters · Streamlit · Plotly · pandas · seaborn

## Getting started

```bash
git clone https://github.com/<your-username>/semantic-ombudsman.git
cd semantic-ombudsman
uv sync
```

Run the Streamlit app:

```bash
uv run streamlit run app_ouvidoria.py
```

Open the notebooks:

```bash
uv run jupyter lab
```

The embedding models are downloaded from the Hugging Face Hub on first use, so an internet connection is required the first time.

## App features

- **🔍 Semantic search:** describe a problem and get the top-k most similar complaints, color-coded by score (🟢 > 0.7, 🟡 > 0.5, 🔴 otherwise).
- **📋 Full base:** table with all complaints and a similarity matrix heatmap.
- **🔴 Vector space:** 2D projection (PCA or t-SNE) colored by official category.
- **🔴 Chunking:** paste a long complaint, choose the splitter and parameters, and inspect the chunks and their embeddings.

The sidebar lets you choose the embedding model and the number of results (top-k).

## Project structure

```
semantic-ombudsman/
├── data/
│   └── manifestacoes.json
├── notebooks/
│   ├── 001_analise_comparativa.ipynb
│   ├── 002_deteccao_duplicatas.ipynb
│   └── 03_chunking_manifestacoes.ipynb
├── reports/
│   └── RELATORIO.pdf
├── app_ouvidoria.py
├── pyproject.toml
└── README.md
```

## Dataset

`data/manifestacoes.json` contains 40 anonymized complaints (fields: `id`, `data`, `categoria_oficial`, `texto`) across five categories: infrastructure, health, safety, education and environment. About 15% are semantic duplicates and 5 are long texts used for chunking.

## Reference

Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*