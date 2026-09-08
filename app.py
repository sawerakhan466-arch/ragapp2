import os
import re
import hashlib

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .info-box {
            padding: 18px;
            border-radius: 12px;
            background-color: #f5f7fb;
            border: 1px solid #e2e6ef;
            margin-bottom: 15px;
        }

        .source-box {
            padding: 12px;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #4f46e5;
            margin-top: 8px;
        }

        .answer-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        .metric-box {
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            background-color: #f5f7fb;
            border: 1px solid #e2e6ef;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# ============================================================
# SESSION STATE
# ============================================================

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "metadata" not in st.session_state:
    st.session_state.metadata = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from every page of the uploaded PDF.

    Returns:
        pages: list of dictionaries containing page number and text
    """

    pages = []

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(pdf_document, start=1):
        text = page.get_text("text")

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    pdf_document.close()

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean unnecessary whitespace while preserving readable text.
    """

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split extracted PDF text into overlapping chunks.

    Each chunk keeps its source page number.
    """

    chunks = []
    metadata = []

    for page_data in pages:

        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(chunk_text)

                metadata.append(
                    {
                        "page": page_number
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks, metadata


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def build_faiss_index(chunks, model):
    """
    Convert chunks into embeddings and create a FAISS
    cosine-similarity index using normalized vectors.
    """

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# SEARCH DOCUMENT
# ============================================================

def search_document(question, index, chunks, metadata, model, top_k=TOP_K):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        results.append(
            {
                "text": chunks[idx],
                "page": metadata[idx]["page"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = None

    # Streamlit Cloud secrets
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]

    # Optional local environment variable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, retrieved_documents):

    client = get_groq_client()

    if client is None:
        return (
            "⚠️ GROQ_API_KEY is not configured. "
            "Please add your Groq API key in Streamlit Cloud Secrets."
        )

    context_parts = []

    for i, document in enumerate(retrieved_documents, start=1):

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {document['page']}

{document['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY using the HR policy
information provided in the retrieved context.

Rules:

1. Do not invent HR policies.
2. Do not use outside knowledge when answering policy questions.
3. If the answer is not available in the context, clearly say:
   "I could not find this information in the uploaded HR policy."
4. Give clear and professional answers.
5. When possible, mention the relevant policy page.
6. If the policy contains conditions, exceptions, eligibility
   requirements, or approval requirements, include them.
7. Never claim something is allowed or prohibited unless the
   retrieved policy supports that conclusion.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question using only the HR Policy Context.
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,
            max_tokens=1200
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"❌ Groq API error: {str(e)}"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📘 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload an HR policy PDF and ask questions using
    Retrieval-Augmented Generation (RAG).
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        f"**LLM:** `{GROQ_MODEL}`"
    )

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL_NAME}`"
    )

    st.write(
        f"**Retrieval:** Top {TOP_K} chunks"
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "HR Policy Assistant uses RAG to retrieve relevant "
        "sections from your uploaded policy document."
    )


# ============================================================
# API KEY STATUS
# ============================================================

client = get_groq_client()

if client is None:

    st.warning(
        "⚠️ Groq API key is not configured. "
        "The document can still be indexed, but questions "
        "cannot be answered until GROQ_API_KEY is added."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"],
    help="Upload the HR policy document you want the assistant to answer questions from."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(pdf_bytes).hexdigest()

    if current_hash != st.session_state.document_hash:

        with st.spinner("📖 Reading HR policy PDF..."):

            pages = extract_text_from_pdf(pdf_bytes)

        if not pages:

            st.error(
                "❌ No readable text was found in this PDF. "
                "Please upload a text-based PDF."
            )

        else:

            with st.spinner("✂️ Creating text chunks..."):

                chunks, metadata = create_chunks(pages)

            with st.spinner("🧠 Creating embeddings and FAISS index..."):

                embedding_model = load_embedding_model()

                index = build_faiss_index(
                    chunks,
                    embedding_model
                )

            st.session_state.document_hash = current_hash
            st.session_state.chunks = chunks
            st.session_state.metadata = metadata
            st.session_state.index = index
            st.session_state.messages = []

            st.success(
                f"✅ HR policy processed successfully. "
                f"{len(pages)} pages and {len(chunks)} chunks indexed."
            )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.index is not None:

    st.subheader("📊 Document Status")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-box">
                <h3>{len(st.session_state.metadata)}</h3>
                <p>Pages</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-box">
                <h3>{len(st.session_state.chunks)}</h3>
                <p>Text Chunks</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-box">
                <h3>FAISS</h3>
                <p>Vector Search</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Page {source['page']}</b><br>
                            Similarity: {source['score']:.3f}<br><br>
                            {source['text']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about the HR policy..."
)


if question:

    if st.session_state.index is None:

        st.warning(
            "Please upload an HR Policy PDF before asking questions."
        )

    else:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Retrieve relevant chunks
        with st.spinner("🔎 Searching the HR policy..."):

            embedding_model = load_embedding_model()

            retrieved_documents = search_document(
                question=question,
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                metadata=st.session_state.metadata,
                model=embedding_model,
                top_k=TOP_K
            )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("🤖 Generating answer..."):

                answer = generate_answer(
                    question,
                    retrieved_documents
                )

            st.markdown(answer)

            with st.expander("📚 View Retrieved Sources"):

                for source in retrieved_documents:

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Page {source['page']}</b><br>
                            Similarity Score:
                            {source['score']:.3f}<br><br>
                            {source['text']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved_documents
            }
        )
