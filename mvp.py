#!/usr/bin/env python3
# =====================================================================================
# Oxford AQA ESL Marker (Agentic AI MVP) — Vision-only (no Poppler/Tesseract)
# -------------------------------------------------------------------------------------
# • PDFs → images with PyMuPDF (fitz). No Poppler. No Tesseract.
# • Vision LLM (gpt-4o/4.1/5) reads page images to extract rubric / student Q&A.
# • Text LLM marks each question using extracted rubric slices + general criteria.
# • Thread-safe worker (no Streamlit calls inside threads).
# • Detailed terminal logs, progress in UI, auto-refresh while jobs run.
# • PDF report generation (ReportLab) + JSON download.
#
# Requirements:
#   pip install streamlit openai pydantic tenacity pillow pymupdf reportlab
#
# Env:
#   OPENAI_API_KEY=sk-...
#
# Run:
#   streamlit run mvp.py
# =====================================================================================

import io, os, json, re, threading, uuid, time, logging, sys, base64
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from PIL import Image

# Report PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

from dotenv import load_dotenv
import os
load_dotenv() 

# --------------------------------- Logging -------------------------------------------
LOG_LEVEL = os.getenv("MVP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mvp")

# ------------------------------- Streamlit helpers -----------------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"

# -------------------------------- API & Models ---------------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_MOSTAFA")

# Ihab Key
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
SUPPORTED_MODELS = ["gpt-4.1", "gpt-5"]

# --------------------------------- Page & State --------------------------------------
st.set_page_config(page_title="AQA ESL Marker (Vision-only MVP)", layout="wide")

def init_state():
    defaults = {
        "jobs": {},           # id -> JobStatus
        "rubric": None,       # RubricExtract or None
        "llm_model": SUPPORTED_MODELS[0],
        "render_scale": 2.2,  # image resolution: 1.8–2.5 typical
        "verbose": False,
        "_last_refresh": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
log.info("Session keys: %s", list(st.session_state.keys()))

# ---------------------------------- Data Models --------------------------------------
class QuestionExtraction(BaseModel):
    question_number: str
    question_details: str
    student_answer: str

class RubricQuestionCriterion(BaseModel):
    question_number: str
    criteria: List[str] = Field(default_factory=list)
    grade_bands: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None

class RubricExtract(BaseModel):
    general_criteria: List[str] = Field(default_factory=list)
    general_grade_bands: Dict[str, str] = Field(default_factory=dict)
    question_criteria: List[RubricQuestionCriterion] = Field(default_factory=list)
    max_marks: Dict[str, Dict[str, int]] = Field(default_factory=lambda: {
        "1": {"content": 6, "language": 3, "total": 9},
        "2": {"content": 6, "language": 6, "total": 12},
        "3": {"content": 8, "language": 8, "total": 16},
        "4": {"content": 8, "language": 15, "total": 23},
        "total": {"content": 28, "language": 32, "total": 60},
    })

class QuestionMarkResult(BaseModel):
    question_number: str
    band: Optional[str]
    numeric_mark: Optional[float]
    feedback_summary: str
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    needs_teacher_review: bool = False
    review_reason: Optional[str] = None

class PaperMarkResult(BaseModel):
    paper_id: str
    student_full_name: Optional[str] = None
    questions: List[QuestionMarkResult] = Field(default_factory=list)
    total_mark: Optional[float] = None
    needs_teacher_review: bool = False
    review_reason: Optional[str] = None
    extraction_warnings: List[str] = Field(default_factory=list)

# -------------------------------- PDF → Images (PyMuPDF) -----------------------------
def render_pdf_to_images(file_bytes: bytes, scale: float = 2.0) -> List[Image.Image]:
    """
    Render all PDF pages to PIL images using PyMuPDF (fitz). No Poppler required.
    'scale' is a zoom factor (1.0 base). 2.0–2.5 recommended for handwriting clarity.
    """
    import fitz  # PyMuPDF
    log.info("Rendering PDF → images with PyMuPDF (scale=%.2f)…", scale)
    images: List[Image.Image] = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    log.info("PDF pages: %d", doc.page_count)
    mat = fitz.Matrix(scale, scale)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        log.info("Rendered page %d → %dx%d", i, img.width, img.height)
    doc.close()
    return images

def image_to_data_url(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

# ----------------------- Vision prompts (extraction via images) -----------------------
SYSTEM_MARK_SCHEME_EXTRACTOR = """You are a meticulous academic assistant for the Oxford AQA International GCSE ESL 9280/W Writing Paper.
You will receive the mark scheme as PAGE IMAGES. Produce a STRICT JSON object:
{
  "general_criteria": [string, ...],
  "general_grade_bands": { "Level 5": string, "Level 4": string, "Level 3": string, "Level 2": string, "Level 1": string },
  "question_criteria": [
    {
      "question_number": "1|2|3|4",
      "criteria": [string, ...],
      "grade_bands": { "Level X": string, ... },
      "notes": "optional"
    }
  ]
}
Rules:
- Copy wording faithfully; do NOT invent criteria.
- If a band or criterion isn't present, omit it.
- Keep each bullet atomic and concise (≤ 25 words).
- Return ONLY JSON (no extra text).
"""

SYSTEM_PAPER_EXTRACTOR = """You are an ESL exam extraction agent.
Input: FULL student paper as PAGE IMAGES (cover + questions + additional pages).
Goal: (1) Identify student's FULL NAME, (2) Extract the 4 writing questions and the COMPLETE student answer for each.

Return STRICT JSON:
{
  "student_full_name": string|null,
  "questions": [
    {"question_number":"1|2|3|4","question_details":"...","student_answer":"..."}
  ]
}

CRITICAL:
- Include full answer text for each question. If an answer cannot be read, do NOT invent; instead return {"needs_teacher_review": true, "reason": "..."}.
- Do not return empty strings for 'student_answer'; either return the real text or teacher_review.
- Use text visible across all provided page images. Incorporate content from "additional page" sections into the correct question.
- Return ONLY JSON.
"""

QUESTION_MARKER_SYSTEM = """You are an Oxford AQA ESL (9280/W) examiner.
Task: Mark a SINGLE question using BOTH the question-specific rubric and the general rubric.

Return STRICT JSON:
{
  "question_number": "1|2|3|4",
  "band": "Level X" | null,
  "numeric_mark": number | null,
  "feedback_summary": "1–3 sentences",
  "strengths": ["...", "..."],
  "areas_for_improvement": ["...", "..."],
  "needs_teacher_review": boolean,
  "review_reason": "optional string"
}

Rules:
- Always include 'question_number' in the output.
- Select a Level present in the rubric slice when possible (or null if not determinable).
- Provide a numeric mark based on total marks in the mark scheme.
- Be evidence-based: quote or paraphrase exact phrases from the student's answer.
- For areas for improvement, add corrections and what could have made the mistake better.
- If the answer is fragmented/unclear, set needs_teacher_review=true and explain why.
"""

# ------------------------------- LLM Utilities ---------------------------------------
def parse_json_safely(s: str):
    s = s.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        alt = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if alt:
            try:
                return json.loads(alt.group(1))
            except Exception:
                pass
    raise ValueError("Could not parse JSON from model output.")

def _model() -> str:
    m = st.session_state.get("llm_model", SUPPORTED_MODELS[0])
    return m if m in SUPPORTED_MODELS else SUPPORTED_MODELS[0]

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,))
)
def llm_chat(messages, temperature=0.1):
    model = _model()
    log.info("LLM call -> model=%s, temp=%.2f, parts=%d", model, temperature, len(messages))
    resp = client.chat.completions.create(model=model, temperature=temperature, messages=messages)
    content = resp.choices[0].message.content
    log.info("LLM call OK: %d chars", len(content or ""))
    return content

def vision_messages_from_images(system_prompt: str, images: List[Image.Image], user_text: Optional[str] = None):
    parts = []
    if user_text:
        parts.append({"type": "text", "text": user_text})
    for i, img in enumerate(images, start=1):
        url = image_to_data_url(img, fmt="PNG")
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parts},
    ]

# --------------- Vision → JSON: rubric & paper (name + Q/answers) --------------------
def vision_extract_rubric_from_images(images: List[Image.Image]) -> RubricExtract:
    log.info("[Rubric] Vision extraction from %d page image(s)…", len(images))
    out = llm_chat(vision_messages_from_images(SYSTEM_MARK_SCHEME_EXTRACTOR, images), temperature=0.1)
    data = parse_json_safely(out)
    rub = RubricExtract(**data)
    log.info("[Rubric] Parsed rubric: %d general criteria, %d question slices",
             len(rub.general_criteria), len(rub.question_criteria))
    return rub

def vision_extract_paper_from_images(images: List[Image.Image]) -> Tuple[Optional[str], List[QuestionExtraction], Optional[str]]:
    log.info("[Extract] Vision extraction (student name + questions) from %d image(s)…", len(images))
    out = llm_chat(vision_messages_from_images(SYSTEM_PAPER_EXTRACTOR, images), temperature=0.0)
    data = parse_json_safely(out)
    if isinstance(data, dict) and data.get("needs_teacher_review"):
        log.warning("[Extract] Needs teacher review: %s", data.get("reason"))
        return None, [], data.get("reason")
    student_name = data.get("student_full_name")
    qarr = data.get("questions", [])
    try:
        questions = [QuestionExtraction(**x) for x in qarr]
    except Exception as e:
        log.exception("[Extract] JSON schema error: %s", e)
        return student_name, [], "Schema error in extraction JSON."
    for q in questions:
        ans = (q.student_answer or "").strip()
        log.info("[Extract] Q%s answer length: %d | Preview: %r", q.question_number, len(ans), ans[:200])
    return student_name, questions, None

# -------------------------------- Marking (text-only) --------------------------------
def llm_mark_question(q: QuestionExtraction, rub: RubricExtract) -> QuestionMarkResult:
    q_r = next((r for r in rub.question_criteria if r.question_number.strip() == q.question_number.strip()), None)
    payload = {
        "question_number": q.question_number,
        "question_details": q.question_details,
        "student_answer": q.student_answer,
        "rubric": {
            "question_specific": (q_r.model_dump() if q_r else {}),
            "general": {
                "general_criteria": rub.general_criteria,
                "general_grade_bands": rub.general_grade_bands,
            },
            "max_marks": rub.max_marks.get(q.question_number, {})
        }
    }
    user = json.dumps(payload, ensure_ascii=False)
    log.info("[Mark Q%s] Sending to LLM (payload %d chars)", q.question_number, len(user))
    out = llm_chat(
        messages=[
            {"role": "system", "content": QUESTION_MARKER_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2
    )
    data = parse_json_safely(out)
    if isinstance(data, dict) and "question_number" not in data:
        data["question_number"] = q.question_number
    try:
        return QuestionMarkResult(**data)
    except ValidationError as e:
        log.exception("[Mark Q%s] Validation failed: %s", q.question_number, e)
        return QuestionMarkResult(
            question_number=q.question_number,
            band=None, numeric_mark=None,
            feedback_summary="Automatic marking failed.",
            strengths=[], areas_for_improvement=[],
            needs_teacher_review=True,
            review_reason="Validation error on marker JSON."
        )

# ------------------------------ PDF Report Generator ---------------------------------
def build_pdf_report(res: PaperMarkResult) -> bytes:
    """
    Build a clean PDF report using ReportLab and return bytes.
    Includes student name, total mark, per-question marks & feedback.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=12)
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], leading=14)

    story = []
    story.append(Paragraph("ESL Writing Marking Report", title))
    story.append(Paragraph(f"Paper ID: <b>{res.paper_id}</b>", body))
    story.append(Paragraph(f"Student: <b>{res.student_full_name or '—'}</b>", body))
    story.append(Paragraph(f"Total Mark: <b>{res.total_mark if res.total_mark is not None else '—'}</b>", body))
    if res.needs_teacher_review:
        story.append(Paragraph(f"<font color='red'><b>Teacher Review Required:</b> {res.review_reason or ''}</font>", body))
    story.append(Spacer(1, 12))

    # Table of question marks
    data = [["Question", "Level", "Mark"]]
    for q in res.questions:
        data.append([q.question_number, q.band or "—", q.numeric_mark if q.numeric_mark is not None else "—"])
    tbl = Table(data, colWidths=[80, 140, 100])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (2,1), (2,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Per question details
    for q in res.questions:
        story.append(Paragraph(f"Question {q.question_number}", h2))
        story.append(Paragraph(f"<b>Level:</b> {q.band or '—'} | <b>Mark:</b> {q.numeric_mark if q.numeric_mark is not None else '—'}", body))
        if q.needs_teacher_review:
            story.append(Paragraph(f"<font color='red'><b>Needs Review:</b> {q.review_reason or ''}</font>", body))
        if q.feedback_summary:
            story.append(Paragraph(f"<b>Feedback Summary:</b> {q.feedback_summary}", body))
        if q.strengths:
            story.append(Paragraph("<b>Strengths:</b>", body))
            for s in q.strengths:
                story.append(Paragraph(f"• {s}", body))
        if q.areas_for_improvement:
            story.append(Paragraph("<b>Areas for Improvement:</b>", body))
            for a in q.areas_for_improvement:
                story.append(Paragraph(f"• {a}", body))
        story.append(Spacer(1, 12))

    doc.build(story)
    return buf.getvalue()

# ------------------------------ Orchestration (thread-safe) ---------------------------
@dataclass
class JobStatus:
    id: str
    filename: str
    status: str
    progress: float
    message: str = ""
    result: Optional[PaperMarkResult] = None

def mark_single_paper_vision(
    fname: str,
    fbytes: bytes,
    rubric: RubricExtract,
    render_scale: float,
    status_cb=lambda **_: None
) -> PaperMarkResult:
    """
    Includes ETA estimation: steps = 2 + N_questions (render, extract, mark each question).
    ETA based on average step duration so far.
    """
    start = time.time()
    completed_steps = 0
    avg_per_step = 2.0  # initial guess (sec), refined as we go

    def set_stage(msg: str, progress: float):
        nonlocal completed_steps, avg_per_step
        elapsed = time.time() - start
        steps_total_guess = completed_steps + 1  # current step in progress
        eta = max(0.0, avg_per_step * max(0, steps_total_guess - completed_steps - 0))  # rough during first step
        status_cb(status="EXTRACTING" if progress < 0.4 else "MARKING" if progress < 1.0 else "DONE",
                  progress=progress,
                  message=f"{msg}")

    pmr = PaperMarkResult(paper_id=str(uuid.uuid4())[:8])

    # 1) PDF → images
    set_stage("Rendering pages…", 0.1)
    log.info("[Job %s] Step 1: Render PDF to images…", pmr.paper_id)
    t0 = time.time()
    images = render_pdf_to_images(fbytes, scale=render_scale)
    step_time = time.time() - t0
    completed_steps += 1
    avg_per_step = (avg_per_step * (completed_steps - 1) + step_time) / completed_steps

    if not images:
        pmr.needs_teacher_review = True
        pmr.review_reason = "Could not render PDF pages."
        status_cb(status="REVIEW", progress=0.95, message="Needs teacher review")
        log.warning("[Job %s] Could not render pages.", pmr.paper_id)
        return pmr

    # 2) Vision extraction for name + Q&As
    set_stage("Extracting name & Q/answers…", 0.25)
    log.info("[Job %s] Step 2: Vision extract (name + Q/answers)…", pmr.paper_id)
    t1 = time.time()
    student_name, questions, extraction_issue = vision_extract_paper_from_images(images)
    step_time = time.time() - t1
    completed_steps += 1
    avg_per_step = (avg_per_step * (completed_steps - 1) + step_time) / completed_steps
    pmr.student_full_name = student_name

    if extraction_issue:
        pmr.needs_teacher_review = True
        pmr.review_reason = extraction_issue
        status_cb(status="REVIEW", progress=0.95, message="Needs teacher review")
        log.warning("[Job %s] Extraction issue: %s", pmr.paper_id, extraction_issue)
        return pmr
    if not questions:
        pmr.needs_teacher_review = True
        pmr.review_reason = "No questions detected."
        status_cb(status="REVIEW", progress=0.95, message="Needs teacher review")
        log.warning("[Job %s] 0 questions extracted.", pmr.paper_id)
        return pmr

    # Estimate total steps for ETA
    total_steps = 2 + len(questions)

    # 3) Marking
    results: List[QuestionMarkResult] = []
    for i, q in enumerate(questions, start=1):
        # progress for this question
        base = 0.4
        frac = 0.55 * (i / len(questions))
        remaining_steps = total_steps - completed_steps
        eta = max(0.0, avg_per_step * remaining_steps)
        status_cb(status="MARKING", progress=base + frac, message=f"Marking Q{q.question_number}…")

        log.info("[Job %s] Mark Q%s (ans len=%d)…", pmr.paper_id, q.question_number, len((q.student_answer or '').strip()))
        t2 = time.time()
        try:
            res = llm_mark_question(q, rub=rubric)
        except Exception as e:
            log.exception("[Job %s] Marking error on Q%s: %s", pmr.paper_id, q.question_number, e)
            res = QuestionMarkResult(
                question_number=q.question_number,
                band=None, numeric_mark=None,
                feedback_summary="Automatic marking failed.",
                strengths=[], areas_for_improvement=[],
                needs_teacher_review=True,
                review_reason=str(e),
            )
        results.append(res)
        step_time = time.time() - t2
        completed_steps += 1
        avg_per_step = (avg_per_step * (completed_steps - 1) + step_time) / completed_steps

    pmr.questions = results
    numeric = [r.numeric_mark for r in results if isinstance(r.numeric_mark, (int, float))]
    pmr.total_mark = round(sum(numeric) / len(numeric), 2) if numeric else None

    if any(r.needs_teacher_review for r in results):
        pmr.needs_teacher_review = True
        reasons = [f"Q{r.question_number}: {r.review_reason}" for r in results if r.needs_teacher_review and r.review_reason]
        pmr.review_reason = "; ".join(reasons) if reasons else "One or more questions need teacher review."
        status_cb(status="REVIEW", progress=0.98, message="Needs teacher review")
        log.warning("[Job %s] Marking flagged items: %s", pmr.paper_id, pmr.review_reason)
    else:
        status_cb(status="DONE", progress=1.0, message="Completed")
        log.info("[Job %s] DONE. Total mark=%s, Student=%s", pmr.paper_id, pmr.total_mark, pmr.student_full_name)

    return pmr

# ------------------------------------------ UI ---------------------------------------
st.sidebar.title("⚙️ Configuration")
st.session_state["llm_model"] = st.sidebar.selectbox(
    "Vision/Text Model", SUPPORTED_MODELS,
    index=SUPPORTED_MODELS.index(st.session_state.get("llm_model", SUPPORTED_MODELS[0]))
)
st.session_state["render_scale"] = st.sidebar.slider("PDF render scale", 1.5, 3.0, float(st.session_state["render_scale"]), step=0.1)
st.session_state["verbose"] = st.sidebar.checkbox("Verbose terminal logs", value=st.session_state.get("verbose", True))
if st.session_state["verbose"]:
    log.setLevel(logging.INFO)
else:
    log.setLevel(logging.WARNING)

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-REPLACE"):
    st.sidebar.warning("Set OPENAI_API_KEY env var or edit the script with your key.")

st.header("Oxford AQA — AI Marker")

# ---- Step 1: Mark scheme (Vision) ----
st.subheader("Step 1: Upload Mark Scheme (PDF)")
mark_pdf = st.file_uploader("Mark Scheme PDF", type=["pdf"], key="mark_scheme")

if st.button("Extract Rubric", type="primary", disabled=mark_pdf is None):
    if mark_pdf:
        st.toast("Rendering + Extracting rubric…", icon="🧠")
        try:
            ms_imgs = render_pdf_to_images(mark_pdf.read(), scale=st.session_state["render_scale"])
            rub = vision_extract_rubric_from_images(ms_imgs)
            st.session_state["rubric"] = rub
            st.success("Rubric extracted successfully.")
        except Exception as e:
            log.exception("Rubric extraction failed: %s", e)
            st.error(f"Rubric extraction failed: {e}")

rub = st.session_state.get("rubric")
if rub:
    with st.expander("Rubric Overview", expanded=False):
        st.markdown("### General Criteria")
        for c in rub.general_criteria:
            st.markdown(f"- {c}")
        st.markdown("### General Grade Levels")
        for b, d in rub.general_grade_bands.items():
            st.markdown(f"**{b}** — {d}")
        st.markdown("### Question Criteria")
        for q in rub.question_criteria:
            st.markdown(f"**Q{q.question_number}:**")
            if q.criteria:
                for c in q.criteria:
                    st.markdown(f"- {c}")
            if q.grade_bands:
                st.markdown("_Levels:_")
                for b, d in q.grade_bands.items():
                    st.markdown(f"- **{b}** — {d}")
            if q.notes:
                st.caption(q.notes)
        st.caption(f"Max marks sanity: {json.dumps(rub.max_marks)}")

st.divider()

# ---- Step 2: Student papers (Vision) ----
st.subheader("Step 2: Upload Student Papers (PDFs)")
papers = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

@dataclass
class JobStatus:
    id: str
    filename: str
    status: str
    progress: float
    message: str = ""
    result: Optional[PaperMarkResult] = None

def launch_job(fname: str, fbytes: bytes):
    rubric_snapshot: Optional[RubricExtract] = st.session_state.get("rubric")
    scale = float(st.session_state["render_scale"])

    jid = str(uuid.uuid4())[:8]
    job = JobStatus(jid, fname, "QUEUED", 0.0, "Queued")

    jobs = st.session_state.get("jobs", {})
    jobs[jid] = job
    st.session_state["jobs"] = jobs

    def update(**kw):
        for k, v in kw.items():
            setattr(job, k, v)

    def worker():
        log.info("[Job %s] START → %s", job.id, job.filename)
        try:
            if not rubric_snapshot:
                update(status="ERROR", progress=1.0, message="No rubric loaded. Extract rubric first.")
                log.error("[Job %s] Aborted: no rubric.", job.id)
                return
            update(status="EXTRACTING", progress=0.05, message="Rendering pages…")
            res = mark_single_paper_vision(
                fname=fname,
                fbytes=fbytes,
                rubric=rubric_snapshot,
                render_scale=scale,
                status_cb=update
            )
            update(result=res)
            if res.needs_teacher_review and res.review_reason:
                update(status="REVIEW", progress=0.98, message="Needs teacher review")
            else:
                update(status="DONE", progress=1.0, message="Completed")
            log.info("[Job %s] FINISH. Status=%s", job.id, job.status)
        except Exception as e:
            update(status="ERROR", progress=1.0, message=f"Error: {e}")
            log.exception("[Job %s] CRASH: %s", job.id, e)

    threading.Thread(target=worker, daemon=True).start()

if st.button("Submit for Marking", type="primary", disabled=not papers):
    if not st.session_state.get("rubric"):
        st.warning("Please extract the rubric first.")
    else:
        for p in papers:
            launch_job(p.name, p.read())
        st.success(f"Submitted {len(papers)} paper(s).")

st.divider()

# ---- Step 3: Queue & results ----
# ---- Step 3: Marking Queue & Results
st.subheader("Step 3: Marking Queue & Results")

jobs_dict: Dict[str, JobStatus] = st.session_state.get("jobs", {})
if not jobs_dict:
    st.info("No active jobs.")
else:
    # 1) Compute running flag first
    running = any(j.status in ("QUEUED", "EXTRACTING", "MARKING") for j in jobs_dict.values())

    # 2) RENDER the job list (progress bars & details)
    for jid, job in list(jobs_dict.items()):
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.markdown(f"**{job.filename}**")
            # ensure value is clipped [0, 1]
            prog_val = job.progress if isinstance(job.progress, (int, float)) else 0.0
            prog_val = max(0.0, min(1.0, prog_val))
            c2.progress(prog_val, text=(job.message or ""))  # shows progress bar
            c3.markdown(f"**{job.status}**")
            if job.result and job.result.student_full_name:
                c4.markdown(f"👤 {job.result.student_full_name}")

            if job.result:
                res = job.result
                st.markdown(f"**Total Mark:** {res.total_mark if res.total_mark is not None else '—'}")
                if res.needs_teacher_review:
                    st.warning(f"Teacher Review: {res.review_reason or 'Required'}")

                for q in res.questions:
                    with st.expander(f"Question {q.question_number}", expanded=False):
                        st.markdown(f"**Level:** {q.band or '—'} | **Mark:** {q.numeric_mark if q.numeric_mark is not None else '—'}")
                        if q.needs_teacher_review:
                            st.warning(q.review_reason or "Needs review")
                        st.markdown(f"**Feedback:** {q.feedback_summary or '—'}")
                        if q.strengths:
                            st.markdown("**Strengths:**")
                            for s in q.strengths:
                                st.markdown(f"- {s}")
                        if q.areas_for_improvement:
                            st.markdown("**Areas for improvement:**")
                            for a in q.areas_for_improvement:
                                st.markdown(f"- {a}")

                # downloads...
                report_json = json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
                st.download_button(
                    "Email Student Report",
                    data=report_json,
                    file_name=f"{job.result.student_full_name}.report.json",
                    mime="application/json"
                )
                try:
                    pdf_bytes = build_pdf_report(res)
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{job.result.student_full_name}.report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    log.exception("PDF build failed: %s", e)
                    st.error(f"PDF build failed: {e}")

    # 3) AFTER rendering, trigger auto-refresh (so bars stay live)
    if running:
        # st.caption("⏳ Live updating… (auto-refresh ~1.5s)")
        time.sleep(5)
        safe_rerun()
