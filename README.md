# DellBERT | Review Insights – README
A topic and sentiment analysis classifier for costumer reviews.

*(Pure-ML pipeline: preprocessing ➜ topic modeling ➜ sentiment classifier ➜ summaries)*

---

## 1. What this repo / notebook delivers

| Stage                         | Output                                                                               | Purpose                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **01 Clean.ipynb**            | `clean_reviews.csv`                                                                  | English-only, emoji-free review text + star rating + 3-class sentiment label                   |
| **03 Topics.ipynb**           | `with_topics.csv`                                                                    | 10-topic LDA model + human-readable KeyBERT labels; each review tagged with its dominant topic |
| **05 Sentiment.ipynb**        | `sentiment_final/`                                                                   | DistilBERT-base fine-tuned via LoRA (3-class) + tokenizer                                      |
| **06 Aggregate.ipynb**        | `topic_sentiment_summary.csv`                                                        | Support-ready pivot: topic × {negative, neutral, positive, total}                              |
| **Demo-cell** (README bottom) | live predictions                                                                     | Classify any new review text → topic + sentiment                                               |

---

## 2. Quick-start (Google Colab)

1. Open **`00_setup.ipynb`** → click *Run all* (installs packages).
2. Upload raw reviews CSV (`FusionTech Online Reviews Data Set.csv`).
3. Execute notebooks **in numeric order**.
   *Run-all runtime ≈ 10 min on a free T4.*
4. At the end of **06 Aggregate**, download `topic_sentiment_summary.csv`.

---

## 3. Local setup (GPU workstation / server)

```bash
git clone <this-repo>
cd FusionTech-Review-Insights
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # transformers, datasets, peft, scikit-learn, keybert …
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

*Rough GPU memory guide*

| Card              | Pipeline fits? | Notes                    |
| ----------------- | -------------- | ------------------------ |
| T4 / A10 24 GB    | **Yes**        | LoRA adapter + inference |
| A100 / H100 80 GB | **Training**   | Full CV loop             |

---

## 4. Running the end-to-end demo script

```bash
python demo_predict.py \
   --model_dir sentiment_final \
   --vectorizer models/lda_vectorizer.joblib \
   --lda_model models/lda_model.joblib \
   --labels_pickle models/topic_labels.pkl \
   --input reviews_to_score.txt \
   --out scored_reviews.csv
```

*`scored_reviews.csv` adds columns:* `predicted_topic`, `predicted_sentiment`.

---

## 5. File / folder glossary

| Path                           | What it contains                                           |
| ------------------------------ | ---------------------------------------------------------- |
| `clean_reviews.csv`            | Pre-processed review text + stars + 3-class label          |
| `sentiment_final/`             | DistilBERT checkpoint with LoRA adapter + tokenizer        |
| `with_topics.csv`              | Every review with `topic_id` column already mapped to text |
| `topic_sentiment_summary.csv`  | Pivot table for Customer-Support triage                    |
| `demo_predict.py`              | CLI example (loads all artefacts, scores new text)         |

---

## 6. Updating the model with fresh data

1. Append new reviews to the raw CSV.
2. Re-run **01 Clean** → **03 Topics** (optional sub-topics) → **05 Sentiment** (LoRA fine-tune takes ≈ 6 min per epoch) → **06 Aggregate**.
3. Swap the old adapter with the new `sentiment_final/` in production; topic model is drop-in if vocabulary unchanged.

---

## 7. Licence & acknowledgements

* Code: MIT
* Pre-trained model weights: Apache 2.0 via Hugging Face (`distilbert-base-uncased`)
* LDA/KeyBERT rely on scikit-learn BSD 3-Clause & sentence-transformers Apache 2.0

Made with ♥ by the FusionTech ML team.
