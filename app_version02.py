import json
import math
import os
import re
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from google import genai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Bangla Multi-Publisher News Analyzer",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Bangla Multi-Publisher News Analyzer")
st.caption(
    "প্রশ্নের বিষয় শনাক্ত করে একই ঘটনার সব সংবাদমাধ্যমের কভারেজ তুলনা করে।"
)


# =========================================================
# Configuration
# =========================================================
DATASET_URL = (
    "https://huggingface.co/datasets/Bilash911/BanglaNewsDataset/"
    "resolve/main/newspaperDataset.csv"
)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


# =========================================================
# Structured-output models
# =========================================================
class ConsensusItem(BaseModel):
    point: str = Field(description="A factual point supported by most distinct publishers.")
    supporting_publishers: List[str]
    explanation: str


class PublisherPosition(BaseModel):
    position: str
    publishers: List[str]


class DisagreementItem(BaseModel):
    issue: str
    positions: List[PublisherPosition]
    explanation: str


class PublisherAnalysis(BaseModel):
    publisher: str
    article_count: int
    headlines: List[str]
    emphasis: List[str]
    tone: str
    tone_explanation: str
    wording_features: List[str]
    missing_relative_to_others: List[str]


class CoverageGap(BaseModel):
    gap: str
    why_it_matters: str


class ComparativeAnalysis(BaseModel):
    selected_topic: str
    answer: str
    consensus: List[ConsensusItem]
    disagreements: List[DisagreementItem]
    publishers: List[PublisherAnalysis]
    coverage_gaps: List[CoverageGap]
    limitations: List[str]


def pydantic_schema(model_class):
    """Support both Pydantic v1 and v2."""
    if hasattr(model_class, "model_json_schema"):
        return model_class.model_json_schema()
    return model_class.schema()


def parse_pydantic_json(model_class, text: str):
    """Support both Pydantic v1 and v2."""
    if hasattr(model_class, "model_validate_json"):
        return model_class.model_validate_json(text)
    return model_class.parse_raw(text)


# =========================================================
# Dataset loading and cleaning
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_news_data() -> pd.DataFrame:
    dataset = load_dataset(
        "csv",
        data_files=DATASET_URL,
        split="train",
    )
    df = dataset.to_pandas()

    # Normalize the original CSV column names once.
    rename_map = {
        "Incident": "incident",
        "news headline": "headline",
        "news body": "body",
        "news source": "source",
        "news link": "link",
        "news link ": "link",
    }
    df = df.rename(columns=rename_map)

    required_columns = ["incident", "headline", "body", "source", "link"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset columns missing: {missing_columns}")

    df = df[required_columns].dropna(how="all").copy()

    for column in ["incident", "headline", "source", "link"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df = df[
        (df["incident"] != "")
        & (df["headline"] != "")
        & (df["source"] != "")
    ].copy()

    df["body_available"] = df["body"].str.len() >= 80
    df["article_id"] = [f"A{i + 1}" for i in range(len(df))]
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# Topic/incident retrieval
# =========================================================
def make_incident_profiles(df: pd.DataFrame):
    profiles = []
    incidents = []

    for incident, group in df.groupby("incident", sort=True):
        sample_headlines = group["headline"].drop_duplicates().head(12).tolist()
        sample_sources = group["source"].drop_duplicates().head(12).tolist()
        profile = (
            f"ঘটনা: {incident}\n"
            f"শিরোনাম: {' | '.join(sample_headlines)}\n"
            f"সংবাদমাধ্যম: {' | '.join(sample_sources)}"
        )
        incidents.append(incident)
        profiles.append(profile)

    return incidents, profiles


@st.cache_resource(show_spinner=False)
def build_incident_index():
    df = load_news_data()
    incidents, profiles = make_incident_profiles(df)
    model = load_embedding_model()
    embeddings = model.encode(
        [f"passage: {profile}" for profile in profiles],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return incidents, np.asarray(embeddings)


def rank_incidents(query: str, top_k: int = 3) -> pd.DataFrame:
    incidents, incident_embeddings = build_incident_index()
    model = load_embedding_model()
    query_embedding = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = incident_embeddings @ query_embedding
    order = np.argsort(scores)[::-1][:top_k]

    return pd.DataFrame(
        {
            "incident": [incidents[i] for i in order],
            "similarity": [float(scores[i]) for i in order],
        }
    )


# =========================================================
# Context construction
# =========================================================
def build_topic_context(
    topic_df: pd.DataFrame,
    max_chars_per_article: int = 6000,
) -> str:
    incident = topic_df["incident"].iloc[0]
    publisher_count = topic_df["source"].nunique()

    blocks = [
        f"SELECTED INCIDENT: {incident}",
        f"TOTAL ARTICLES: {len(topic_df)}",
        f"DISTINCT PUBLISHERS: {publisher_count}",
        "",
    ]

    # Grouping keeps the model aware that one publisher may have multiple articles.
    for publisher_number, (source, source_group) in enumerate(
        topic_df.groupby("source", sort=True), start=1
    ):
        blocks.append(f"PUBLISHER P{publisher_number}: {source}")
        blocks.append(f"ARTICLE COUNT: {len(source_group)}")

        for article_number, (_, row) in enumerate(source_group.iterrows(), start=1):
            body = row["body"]
            if not row["body_available"]:
                body = "[পূর্ণ সংবাদবডি ডেটাসেটে অনুপস্থিত; শুধু শিরোনাম পাওয়া গেছে]"
            else:
                body = body[:max_chars_per_article]

            blocks.extend(
                [
                    f"ARTICLE {publisher_number}.{article_number}",
                    f"Headline: {row['headline']}",
                    f"Link: {row['link']}",
                    f"Body: {body}",
                    "---",
                ]
            )
        blocks.append("=" * 70)

    return "\n".join(blocks)


def create_comparison_prompt(
    user_question: str,
    incident: str,
    topic_df: pd.DataFrame,
    context: str,
) -> str:
    publishers = sorted(topic_df["source"].unique().tolist())
    publisher_count = len(publishers)
    consensus_minimum = max(2, math.ceil(publisher_count / 2))

    return f"""
You are a rigorous comparative Bangla news-analysis engine.

USER QUESTION:
{user_question}

SELECTED INCIDENT:
{incident}

KNOWN PUBLISHERS ({publisher_count}):
{json.dumps(publishers, ensure_ascii=False)}

Use ONLY the supplied corpus. Do not use outside knowledge.
Write every natural-language field in Bangla.

Definitions and rules:
1. First answer the user's actual question in the `answer` field.
2. A `consensus` item must be a concrete factual claim supported by at least
   {consensus_minimum} distinct publishers. List the exact publisher names.
3. A `disagreement` requires incompatible facts, numbers, causes, timelines,
   responsibility claims, or interpretations. Mere difference of emphasis is
   not a disagreement.
4. `emphasis` means what a publisher gave the most attention to.
5. `wording_features` should describe framing, labels, loaded/neutral wording,
   attribution, certainty, and recurring phrases. Use only very short excerpts.
6. `tone` is a textual tone assessment, not a claim that a publisher is honest,
   dishonest, biased, or unbiased.
7. `missing_relative_to_others` may contain only important facts mentioned by
   at least two other publishers but absent from that publisher's supplied articles.
8. `coverage_gaps` means important questions left unanswered by the entire
   supplied corpus. Phrase these as unanswered questions, not as outside facts.
9. A publisher can have several articles. Produce exactly one consolidated
   `PublisherAnalysis` row for each distinct publisher in the corpus.
10. If an article body is missing, do not infer its content from the headline.
11. Do not invent dates, facts, motives, quotations, or publisher positions.
12. Mention corpus limitations in `limitations`, including missing article bodies
    and the fact that the analysis covers only the supplied dataset.

CORPUS:
{context}
"""


# =========================================================
# Gemini structured analysis
# =========================================================
def strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def analyze_with_gemini(prompt: str) -> ComparativeAnalysis:
    client = load_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY পাওয়া যায়নি। Streamlit Secrets-এ key যোগ করুন।"
        )

    schema = pydantic_schema(ComparativeAnalysis)

    # Current Google GenAI SDK structured-output format.
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": schema,
                    }
                },
            },
        )
        return parse_pydantic_json(ComparativeAnalysis, strip_json_fence(response.text))

    # Compatibility fallback for older google-genai SDK releases.
    except Exception as modern_error:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            return parse_pydantic_json(
                ComparativeAnalysis,
                strip_json_fence(response.text),
            )
        except Exception as legacy_error:
            raise RuntimeError(
                "Gemini structured-output request failed. "
                f"Current-format error: {modern_error}; "
                f"fallback error: {legacy_error}"
            ) from legacy_error


# =========================================================
# Result validation and table rendering
# =========================================================
def validate_analysis(
    analysis: ComparativeAnalysis,
    topic_df: pd.DataFrame,
) -> ComparativeAnalysis:
    """Remove unsupported publisher names and enforce the consensus threshold."""
    valid_publishers = set(topic_df["source"].unique().tolist())
    minimum = max(2, math.ceil(len(valid_publishers) / 2))

    valid_consensus = []
    for item in analysis.consensus:
        item.supporting_publishers = list(
            dict.fromkeys(
                publisher
                for publisher in item.supporting_publishers
                if publisher in valid_publishers
            )
        )
        if len(item.supporting_publishers) >= minimum:
            valid_consensus.append(item)
    analysis.consensus = valid_consensus

    for disagreement in analysis.disagreements:
        for position in disagreement.positions:
            position.publishers = list(
                dict.fromkeys(
                    publisher
                    for publisher in position.publishers
                    if publisher in valid_publishers
                )
            )
        disagreement.positions = [
            position for position in disagreement.positions if position.publishers
        ]
    analysis.disagreements = [
        disagreement
        for disagreement in analysis.disagreements
        if len(disagreement.positions) >= 2
    ]

    return analysis


def join_items(items: List[str]) -> str:
    return "\n".join(f"• {item}" for item in items) if items else "—"


def render_analysis(analysis: ComparativeAnalysis, topic_df: pd.DataFrame):
    st.subheader("উত্তর")
    st.markdown(analysis.answer)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "যেখানে একমত",
            "যেখানে ভিন্নমত",
            "প্রকাশকভিত্তিক বিশ্লেষণ",
            "কী অনুপস্থিত",
            "ব্যবহৃত সংবাদ",
        ]
    )

    with tab1:
        if analysis.consensus:
            consensus_rows = [
                {
                    "একমতের বিষয়": item.point,
                    "প্রকাশক সংখ্যা": len(item.supporting_publishers),
                    "যেসব প্রকাশক": ", ".join(item.supporting_publishers),
                    "ব্যাখ্যা": item.explanation,
                }
                for item in analysis.consensus
            ]
            st.dataframe(
                pd.DataFrame(consensus_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("নির্ধারিত সংখ্যাগরিষ্ঠতার শর্ত পূরণ করা স্পষ্ট ঐকমত্য পাওয়া যায়নি।")

    with tab2:
        if analysis.disagreements:
            disagreement_rows = []
            for item in analysis.disagreements:
                positions = []
                for position in item.positions:
                    positions.append(
                        f"{position.position} — {', '.join(position.publishers)}"
                    )
                disagreement_rows.append(
                    {
                        "বিতর্কের বিষয়": item.issue,
                        "বিভিন্ন অবস্থান": "\n".join(positions),
                        "ব্যাখ্যা": item.explanation,
                    }
                )
            st.dataframe(
                pd.DataFrame(disagreement_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("সরবরাহ করা কভারেজে স্পষ্ট তথ্যগত বিরোধ পাওয়া যায়নি।")

    with tab3:
        publisher_rows = [
            {
                "প্রকাশক": item.publisher,
                "নিবন্ধ সংখ্যা": item.article_count,
                "শিরোনাম": join_items(item.headlines),
                "কী গুরুত্ব দিয়েছে": join_items(item.emphasis),
                "টোন": item.tone,
                "টোনের কারণ": item.tone_explanation,
                "শব্দচয়ন/ফ্রেমিং": join_items(item.wording_features),
                "অন্যদের তুলনায় বাদ পড়েছে": join_items(
                    item.missing_relative_to_others
                ),
            }
            for item in analysis.publishers
        ]
        st.dataframe(
            pd.DataFrame(publisher_rows),
            width="stretch",
            hide_index=True,
            height=620,
        )

    with tab4:
        if analysis.coverage_gaps:
            gap_rows = [
                {
                    "পুরো কভারেজে অনুত্তরিত বিষয়": item.gap,
                    "কেন গুরুত্বপূর্ণ": item.why_it_matters,
                }
                for item in analysis.coverage_gaps
            ]
            st.dataframe(
                pd.DataFrame(gap_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("মডেল পুরো করপাসে নির্দিষ্ট কোনো অনুত্তরিত প্রশ্ন শনাক্ত করেনি।")

        if analysis.limitations:
            st.markdown("#### সীমাবদ্ধতা")
            for limitation in analysis.limitations:
                st.markdown(f"- {limitation}")

    with tab5:
        source_table = topic_df[
            ["source", "headline", "body_available", "link"]
        ].copy()
        source_table.columns = [
            "সংবাদমাধ্যম",
            "শিরোনাম",
            "পূর্ণ বডি আছে",
            "লিংক",
        ]
        st.dataframe(
            source_table,
            width="stretch",
            hide_index=True,
            column_config={
                "লিংক": st.column_config.LinkColumn(
                    "লিংক",
                    display_text="সংবাদ খুলুন",
                )
            },
        )


# =========================================================
# Sidebar and app workflow
# =========================================================
try:
    news_df = load_news_data()
except Exception as exc:
    st.error(f"Dataset load error: {exc}")
    st.stop()

with st.sidebar:
    st.header("বিশ্লেষণ সেটিংস")
    st.metric("মোট ঘটনা", news_df["incident"].nunique())
    st.metric("মোট সংবাদ", len(news_df))
    st.metric("সংবাদমাধ্যম", news_df["source"].nunique())

    max_chars = st.slider(
        "প্রতি নিবন্ধের সর্বোচ্চ অক্ষর",
        min_value=2000,
        max_value=12000,
        value=6000,
        step=1000,
        help="বেশি অক্ষর মানে বেশি context, কিন্তু API খরচ ও latency বাড়তে পারে।",
    )

    manual_topic = st.selectbox(
        "প্রয়োজনে ঘটনা নিজে নির্বাচন করুন",
        ["Auto-detect"] + sorted(news_df["incident"].unique().tolist()),
    )

user_question = st.text_area(
    "আপনার প্রশ্ন লিখুন",
    placeholder=(
        "উদাহরণ: জুলাই গণঅভ্যুত্থান নিয়ে সংবাদমাধ্যমগুলো কী বিষয়ে একমত, "
        "কোথায় ভিন্নমত, এবং কারা কী বেশি গুরুত্ব দিয়েছে?"
    ),
    height=120,
)

if st.button("বিশ্লেষণ করুন", type="primary", use_container_width=True):
    if not user_question.strip():
        st.warning("একটি প্রশ্ন লিখুন।")
        st.stop()

    if manual_topic == "Auto-detect":
        with st.spinner("প্রশ্নের বিষয় শনাক্ত করা হচ্ছে..."):
            ranked = rank_incidents(user_question, top_k=3)
        selected_incident = ranked.iloc[0]["incident"]

        st.markdown(f"**স্বয়ংক্রিয়ভাবে নির্বাচিত ঘটনা:** {selected_incident}")
        with st.expander("অন্যান্য সম্ভাব্য ঘটনা ও similarity score"):
            st.dataframe(ranked, width="stretch", hide_index=True)
    else:
        selected_incident = manual_topic
        ranked = None
        st.markdown(f"**নির্বাচিত ঘটনা:** {selected_incident}")

    topic_df = news_df[news_df["incident"] == selected_incident].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("নিবন্ধ", len(topic_df))
    col2.metric("স্বতন্ত্র প্রকাশক", topic_df["source"].nunique())
    col3.metric("বডি অনুপস্থিত", int((~topic_df["body_available"]).sum()))

    with st.spinner("সব প্রকাশকের কভারেজ সাজানো হচ্ছে..."):
        context = build_topic_context(
            topic_df,
            max_chars_per_article=max_chars,
        )
        prompt = create_comparison_prompt(
            user_question=user_question,
            incident=selected_incident,
            topic_df=topic_df,
            context=context,
        )

    try:
        with st.spinner("Gemini তুলনামূলক বিশ্লেষণ তৈরি করছে..."):
            analysis = analyze_with_gemini(prompt)
            analysis = validate_analysis(analysis, topic_df)

        render_analysis(analysis, topic_df)

        with st.expander("Debug: Gemini-কে পাঠানো context দেখুন"):
            st.text(context)

    except Exception as exc:
        st.error(str(exc))
