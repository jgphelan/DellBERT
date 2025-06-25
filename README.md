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
| **Demo-cell**                 | live predictions                                                                     | Classify any new example review text → sentiment                                               |

---

## 2. Quick-start (Google Colab)

1. Open **`00_setup.ipynb`** → click *Run all* (installs packages).
2. Upload raw reviews CSV (`FusionTech Online Reviews Data Set.csv`).
3. Execute notebooks **in numeric order**.
4. At the end of **06 Aggregate**, download `topic_sentiment_summary.csv`.

---

## 3. File / folder glossary

| Path                           | What it contains                                           |
| ------------------------------ | ---------------------------------------------------------- |
| `clean_reviews.csv`            | Pre-processed review text + stars + 3-class label          |
| `sentiment_final/`             | DistilBERT checkpoint with LoRA adapter + tokenizer        |
| `with_topcs_only_index.csv`    | Every review labeled with 'topic_id'                       |
| `with_topics.csv`              | Every review with `topic_id` column already mapped to text |
| `topic_sentiment_summary.csv`  | Pivot table for Customer-Support triage                    |
| `demo_predict.py`              | CLI example (loads all artefacts, scores new text)         |

---

## 4. Updating the model with fresh data

1. Append new reviews to the raw CSV.
2. Re-run **01 Clean** → **03 Topics** → **05 Sentiment** → **06 Aggregate**.
3. Swap the old adapter with the new `sentiment_final/` in production; topic model is drop-in if vocabulary unchanged.

---

## 5. Licence & acknowledgements

* Code: MIT
* Pre-trained model weights: Apache 2.0 via Hugging Face (`distilbert-base-uncased`)
* LDA/KeyBERT rely on scikit-learn BSD 3-Clause & sentence-transformers Apache 2.0

Made with ♥ by the FusionTech ML team.
