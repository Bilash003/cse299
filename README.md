# Simple Three-Model Bangla News RAG

This Streamlit application:

1. Loads `Bilash911/BanglaNewsDataset`.
2. Creates multilingual sentence embeddings.
3. Retrieves the most relevant news articles.
4. Sends the same RAG prompt to Gemini, Qwen, and Gemma.
5. Uses Gemini as the reference answer.
6. Calculates BLEU, BERTScore, and ROUGE for Qwen and Gemma.

## Important terminology

There is no commonly used text-generation metric called **BARD score**.
This project assumes you meant **BERTScore**.

## Setup

Use Python 3.10 or 3.11.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add:

- A Gemini API key
- A Hugging Face access token

Run:

```bash
streamlit run app.py
```

## Hugging Face model access

The Qwen and Gemma calls use Hugging Face hosted inference, so availability can
depend on your Hugging Face account, provider quota, and model access.

For a gated Gemma repository, open its Hugging Face model page and accept the
license before running the app. You can replace either model ID in `.env`.

## Metric interpretation

- BLEU measures token overlap.
- ROUGE measures unigram, bigram, and longest-common-subsequence overlap.
- BERTScore measures semantic similarity using multilingual contextual embeddings.
- Gemini is treated only as a comparison reference, not a guaranteed gold answer.
