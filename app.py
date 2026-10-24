import os
import re
from io import BytesIO

import requests
import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="MediCore AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = """You are MediCore AI, an evidence-aware medical education tutor for medical students.
Teach across foundational sciences and clinical specialties.

CORE RULES
- Use uploaded-document excerpts as evidence for claims attributed to those documents.
- Cite uploaded material inline as [filename, p. N]. Only use citations supplied in the retrieved context.
- Never invent sources, page numbers, diagnostic criteria, lab values, guidelines, or quotations.
- If uploaded documents do not support an answer, say so. You may separately explain general medical knowledge and label it as not verified against the uploaded library.
- Distinguish established knowledge from uncertainty and avoid false precision.
- For disease questions, cover definition, causes, pathophysiology, manifestations, diagnosis, differential diagnosis, treatment principles, complications, and prognosis as relevant.
- For drugs, discuss class, mechanism, indications, major adverse effects, contraindications, interactions, and monitoring when relevant. Do not guess individualized doses.
- For clinical cases, explain a differential rather than claiming certainty from incomplete information.
- For real symptoms that suggest an emergency, advise urgent professional assessment.
- This is educational support, not a substitute for qualified clinical care.
- Do not ask users to upload identifiable patient information.
- Begin with the direct answer. Use clear headings and tables when useful. Follow the requested depth.
"""

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def split_chunks(text, size=1050, overlap=160):
    text = clean(text)
    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size * 0.65:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            result.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return result

def extract_pdf(upload):
    reader = PdfReader(BytesIO(upload.getvalue()))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        for part_num, part in enumerate(split_chunks(text)):
            chunks.append({
                "filename": upload.name,
                "page": page_num,
                "part": part_num,
                "text": part,
            })
    return chunks

def terms(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}|[\u0600-\u06FF]{2,}", text.lower())

def retrieve(question, chunks, limit=5):
    qterms = set(terms(question))
    if not qterms or not chunks:
        return []
    ranked = []
    for chunk in chunks:
        doc_terms = terms(chunk["text"])
        if not doc_terms:
            continue
        counts = {}
        for token in doc_terms:
            counts[token] = counts.get(token, 0) + 1
        overlap = sum(min(counts.get(t, 0), 3) for t in qterms)
        coverage = sum(1 for t in qterms if t in counts)
        # Simple lightweight ranking: no embedding model or heavy vector database required.
        score = overlap + 1.5 * coverage
        if score:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked[:limit]]

def call_gemini(question, context, style):
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    model = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    if not api_key:
        raise RuntimeError("Add GEMINI_API_KEY in Streamlit app Secrets before asking questions.")
    excerpts = []
    for i, chunk in enumerate(context, 1):
        excerpts.append(
            f"EXCERPT {i} — [{chunk['filename']}, p. {chunk['page']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(excerpts) if excerpts else "No excerpts were retrieved from uploaded documents."
    user_prompt = f"""Requested response style: {style}

Uploaded document excerpts:
{context_text}

Student question:
{question}

Answer using the system instructions. Cite supplied excerpts accurately. Clearly distinguish document-grounded claims from general knowledge."""
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"AI API returned {response.status_code}: {detail}")
    data = response.json()
    return data["choices"][0]["message"]["content"]

st.title("🩺 MediCore AI")
st.caption("Your medical study companion · Mobile-friendly · PDF-grounded answers")
st.warning(
    "Educational use only. Do not upload identifiable patient information. "
    "This prototype is not a validated clinical decision-support system."
)

with st.expander("How this works & privacy", expanded=False):
    st.markdown("""
    - Upload PDFs and ask questions about their content.
    - Relevant text is selected with lightweight keyword retrieval.
    - A hosted language model generates the explanation.
    - Uploaded PDFs are held in the current app session; this starter does not create a permanent medical-library database.
    - Your question and retrieved excerpts are sent to the configured AI provider. Do not include personal or identifiable patient data.
    - Free API availability and limits may change. Check the provider's current terms before relying on the service.
    """)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("1. Add your medical resources")
uploads = st.file_uploader(
    "Choose text-based medical PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Scanned/image-only PDFs need OCR, which is not included yet."
)
if st.button("Process PDFs", type="primary", use_container_width=True, disabled=not uploads):
    new_chunks, names, errors = [], [], []
    progress = st.progress(0)
    for i, upload in enumerate(uploads):
        try:
            extracted = extract_pdf(upload)
            if not extracted:
                errors.append(f"{upload.name}: no extractable text; it may be scanned.")
            else:
                new_chunks.extend(extracted)
                names.append(upload.name)
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")
        progress.progress((i + 1) / len(uploads))
    st.session_state.chunks = new_chunks
    st.session_state.file_names = names
    if new_chunks:
        st.success(f"Processed {len(names)} PDF(s) into {len(new_chunks)} searchable text passages.")
    for err in errors:
        st.warning(err)

if st.session_state.file_names:
    st.caption("Current session library: " + ", ".join(st.session_state.file_names))
else:
    st.caption("No PDFs loaded yet. You can still ask general medical education questions.")

st.divider()
st.subheader("2. Ask MediCore")
style = st.selectbox(
    "Answer style",
    [
        "Detailed medical-student explanation",
        "Concise exam answer",
        "Advanced clinical reasoning",
        "Compare diseases in a table",
        "Flashcards / active recall",
    ],
)
question = st.text_area(
    "Your question",
    placeholder="Example: Explain nephrotic syndrome pathophysiology, diagnostic findings, and how it differs from nephritic syndrome.",
    height=110,
)
source_count = st.slider("Source passages to include", min_value=2, max_value=7, value=4)

if st.button("Ask MediCore", type="primary", use_container_width=True, disabled=not question.strip()):
    with st.spinner("Finding relevant passages and preparing your answer…"):
        try:
            hits = retrieve(question, st.session_state.chunks, limit=source_count)
            answer = call_gemini(question, hits, style)
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "hits": hits,
            })
        except requests.Timeout:
            st.error("The request timed out. Try a shorter question or try again later.")
        except requests.RequestException as exc:
            st.error(f"Network error while contacting the AI service: {exc}")
        except Exception as exc:
            st.error(str(exc))

for item in reversed(st.session_state.history):
    st.markdown("---")
    st.markdown("### " + item["question"])
    st.markdown(item["answer"])
    with st.expander("View retrieved PDF passages"):
        if not item["hits"]:
            st.info("No matching passages were found. The answer may rely on general model knowledge, not your uploaded PDFs.")
        for hit in item["hits"]:
            st.markdown(f"**{hit['filename']} · page {hit['page']}**")
            st.write(hit["text"])

with st.expander("Medical study tools — planned next"):
    st.markdown("""
    - Structured disease profiles
    - Drug monographs and comparison tables
    - Exam question generator
    - Flashcard review
    - Clinical case reasoning
    - Better PDF tables, OCR, and citation verification
    - Persistent personal library with appropriate privacy controls
    """)
