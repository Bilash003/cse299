import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from bert_score import BERTScorer
from datasets import load_dataset
from google import genai
from huggingface_hub import InferenceClient
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Bangla News RAG Chatbot",
    page_icon="📰",
    layout="wide",
)

DATASET_NAME = "Bilash911/BanglaNewsDataset"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_GEMMA_MODEL = "google/gemma-3n-E4B-it"


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default)).strip()


# -----------------------------
# Dataset and retrieval
# -----------------------------
@st.cache_resource(show_spinner="Loading dataset and creating the search index...")
def load_rag_index() -> Tuple[pd.DataFrame, TfidfVectorizer, object]:
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    required_columns = [
        "Incident",
        "news headline",
        "news body",
        "news source",
        "news link",
    ]
    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    df = df[required_columns].fillna("").copy()

    # Keep one searchable document per news article.
    df["document"] = (
        df["Incident"].astype(str)
        + " "
        + df["news headline"].astype(str)
        + " "
        + df["news body"].astype(str)
        + " "
        + df["news source"].astype(str)
    )

    # Remove rows with no useful text.
    df = df[df["document"].str.len() > 40].reset_index(drop=True)

    # Character n-gram TF-IDF works well for Bangla text and is very fast for 260 rows.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    document_matrix = vectorizer.fit_transform(df["document"].tolist())
    return df, vectorizer, document_matrix


def retrieve_documents(
    question: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    document_matrix: object,
    top_k: int,
) -> pd.DataFrame:
    query_vector = vectorizer.transform([question])
    similarities = (document_matrix @ query_vector.T).toarray().ravel()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]
    return results


def build_prompt(question: str, retrieved: pd.DataFrame) -> str:
    context_parts: List[str] = []

    for i, (_, row) in enumerate(retrieved.iterrows(), start=1):
        body = str(row["news body"]).strip()
        # Limit each article so API prompts remain quick and inexpensive.
        if len(body) > 2800:
            body = body[:2800] + "..."

        context_parts.append(
            f"Article {i}\n"
            f"Incident: {row['Incident']}\n"
            f"Headline: {row['news headline']}\n"
            f"Body: {body}\n"
            f"Source: {row['news source']}\n"
            f"Link: {row['news link']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a question-answering assistant for a Bangla news dataset.
Answer the user's question only from the supplied news context.
Write the answer in Bangla because the dataset and questions are Bangla.
Be concise, factual, and do not invent missing information.
When useful, mention the news source names in the answer.
If the answer is not present in the context, clearly say that the dataset does not contain enough information.

NEWS CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()


# -----------------------------
# Model calls
# -----------------------------
def generate_gemini(prompt: str, api_key: str, model_name: str) -> str:
    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=model_name,
        input=prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens":1200,
            "thinking_level": "low",
        },
    )
    answer = getattr(interaction, "output_text", "")
    return str(answer).strip()
# def generate_gemini(
#     prompt: str,
#     api_key: str,
#     model_name: str,
# ) -> str:
#     client = genai.Client(api_key=api_key)

#     response = client.models.generate_content(
#         model=model_name,
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             system_instruction=(
#                 "Use only the supplied news context. "
#                 "Answer the question completely in Bangla. "
#                 "Write 3 to 6 complete sentences. "
#                 "Do not return only source names. "
#                 "Mention source names at the end of the answer."
#             ),
#             max_output_tokens=1200,
#             thinking_config=types.ThinkingConfig(
#                 thinking_level="low"
#             ),
#         ),
#     )

#     answer = (response.text or "").strip()

#     if not answer:
#         raise ValueError("Gemini returned an empty answer.")

#     return answer


def generate_huggingface(
    prompt: str,
    hf_token: str,
    model_name: str,
) -> str:
    client = InferenceClient(provider="auto", api_key=hf_token, timeout=120)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "systemhf_dOChceRRCQyZNZJtHFwtfVGQgKGqKHqviw",
                "content": "Use only the provided context and answer in Bangla.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return str(response.choices[0].message.content).strip()


# -----------------------------
# Evaluationqwen
# -----------------------------
@st.cache_resource(show_spinner="Loading the multilingual BERTScore model...")
def load_bert_scorer() -> BERTScorer:
    # DistilBERT keeps Streamlit memory and startup time lower than full mBERT.
    return BERTScorer(
        model_type="distilbert-base-multilingual-cased",
        num_layers=5,
        rescale_with_baseline=False,
        device="cpu",
    )


def calculate_scores(reference: str, candidate: str) -> Dict[str, float]:
    """Compare one model answer with Gemini as the reference answer."""
    bleu_metric = BLEU(tokenize="none", effective_order=True)
    bleu = bleu_metric.sentence_score(candidate, [reference]).score / 100.0

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    ).score(reference, candidate)

    # BERTScore is the standard semantic metric. "BARD score" is not a standard metric.
    scorer = load_bert_scorer()
    _, _, bert_f1 = scorer.score([candidate], [reference])

    return {
        "BLEU": float(bleu),
        "BERTScore F1": float(bert_f1[0]),
        "ROUGE-1 F1": float(rouge["rouge1"].fmeasure),
        "ROUGE-2 F1": float(rouge["rouge2"].fmeasure),
        "ROUGE-L F1": float(rouge["rougeL"].fmeasure),
    }


def safe_scores(reference: str, candidate: str) -> Tuple[Dict[str, float], str]:
    try:
        return calculate_scores(reference, candidate), ""
    except Exception as exc:
        # Keep the app usable even if BERTScore's model download fails.
        try:
            bleu_metric = BLEU(tokenize="none", effective_order=True)
            bleu = bleu_metric.sentence_score(candidate, [reference]).score / 100.0
            rouge = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=False
            ).score(reference, candidate)
            partial = {
                "BLEU": float(bleu),
                "BERTScore F1": np.nan,
                "ROUGE-1 F1": float(rouge["rouge1"].fmeasure),
                "ROUGE-2 F1": float(rouge["rouge2"].fmeasure),
                "ROUGE-L F1": float(rouge["rougeL"].fmeasure),
            }
            return partial, f"BERTScore could not run: {exc}"
        except Exception as second_exc:
            return {}, f"Evaluation failed: {second_exc}"


# -----------------------------
# User interface
# -----------------------------
st.title("📰 Bangla News RAG Chatbot")
st.caption(
    "Ask one question and receive separate answers from Gemini, Qwen, and Gemma. "
    "Qwen and Gemma are compared against Gemini using BLEU, BERTScore, and ROUGE."
)

with st.sidebar:
    st.header("Settings")

    gemini_api_key = get_secret("GEMINI_API_KEY")
    hf_token = get_secret("HF_TOKEN")

    if not gemini_api_key:
        gemini_api_key = st.text_input("Gemini API key", type="password")
    else:
        st.success("Gemini API key loaded")

    if not hf_token:
        hf_token = st.text_input("Hugging Face token", type="password")
    else:
        st.success("Hugging Face token loaded")

    top_k = st.slider("Retrieved articles", min_value=1, max_value=6, value=3)

    with st.expander("Model IDs"):
        gemini_model = st.text_input(
            "Gemini model",
            value=get_secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        )
        qwen_model = st.text_input(
            "Qwen model",
            value=get_secret("QWEN_MODEL", DEFAULT_QWEN_MODEL),
        )
        gemma_model = st.text_input(
            "Gemma model",
            value=get_secret("GEMMA_MODEL", DEFAULT_GEMMA_MODEL),
        )

    st.info(
        "The first BERTScore calculation downloads a multilingual BERT model, "
        "so that first evaluation can be slower."
    )

try:
    news_df, vectorizer, document_matrix = load_rag_index()
    st.success(f"Dataset ready")
except Exception as exc:
    st.error(f"Could not load the dataset or retrieval index: {exc}")
    st.stop()

question = st.text_area(
    "Ask a question about the dataset",
    placeholder="Example: জুলাই গণঅভ্যুত্থান সম্পর্কে সংবাদগুলো কী বলছে?",
    height=100,
)

ask_button = st.button("Generate three answers", type="primary", use_container_width=True)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not gemini_api_key:
        st.error("Add a Gemini API key in Streamlit secrets or the sidebar.")
        st.stop()
    if not hf_token:
        st.error("Add a Hugging Face token in Streamlit secrets or the sidebar.")
        st.stop()

    retrieved = retrieve_documents(
        question.strip(),
        news_df,
        vectorizer,
        document_matrix,
        top_k,
    )
    prompt = build_prompt(question.strip(), retrieved)

    answers: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    with st.spinner("Generating Gemini answer..."):
        try:
            answers["Gemini"] = generate_gemini(
                prompt, gemini_api_key, gemini_model
            )
        except Exception as exc:
            errors["Gemini"] = str(exc)

    with st.spinner("Generating Qwen answer..."):
        try:
            answers["Qwen"] = generate_huggingface(prompt, hf_token, qwen_model)
        except Exception as exc:
            errors["Qwen"] = str(exc)

    with st.spinner("Generating Gemma answer..."):
        try:
            answers["Gemma"] = generate_huggingface(prompt, hf_token, gemma_model)
        except Exception as exc:
            errors["Gemma"] = str(exc)

    st.subheader("Model answers")
    columns = st.columns(3)
    for column, model_name in zip(columns, ["Gemini", "Qwen", "Gemma"]):
        with column:
            st.markdown(f"### {model_name}")
            if model_name in answers and answers[model_name]:
                st.write(answers[model_name])
            else:
                st.error(errors.get(model_name, "No answer returned."))

    if "Gemini" in answers and answers["Gemini"]:
        evaluation_rows = []
        evaluation_warnings = []

        for candidate_name in ["Qwen", "Gemma"]:
            if candidate_name not in answers or not answers[candidate_name]:
                continue

            with st.spinner(f"Calculating scores for {candidate_name}..."):
                score_values, warning = safe_scores(
                    answers["Gemini"], answers[candidate_name]
                )

            if score_values:
                evaluation_rows.append(
                    {"Candidate": candidate_name, **score_values}
                )
            if warning:
                evaluation_warnings.append(f"{candidate_name}: {warning}")

        if evaluation_rows:
            st.subheader("Evaluation scores")
            score_df = pd.DataFrame(evaluation_rows).set_index("Candidate")
            st.dataframe(
                score_df.style.format("{:.4f}"),
                use_container_width=True,
            )
            st.caption(
                "Gemini is the reference. Higher values indicate greater similarity, "
                "not necessarily greater factual correctness."
            )

        for warning in evaluation_warnings:
            st.warning(warning)
    else:
        st.warning("Scores were skipped because Gemini did not return an answer.")

    st.subheader("Retrieved news sources")
    display_df = retrieved[
        ["news headline", "news source", "news link", "similarity"]
    ].copy()
    display_df.columns = ["Headline", "Source", "Link", "Similarity"]
    st.dataframe(
        display_df.style.format({"Similarity": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Link"),
        },
    )
