import json
import os
from datetime import datetime
from pathlib import Path
from google import genai
from pypdf import PdfReader
import streamlit as st

st.set_page_config(page_title="Econometrics Socratic Tutor", layout="wide")
LOG_FILE = Path("logs/study_log.json")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def find_exam_dirs() -> dict[str, Path]:
  """Locate Exam 1, Exam 2, and Exam 3 folders whether at repo root or inside econ_tutor/."""
  target_names = ["Exam 1", "Exam 2", "Exam 3"]
  candidates = [Path("."), Path("econ_tutor")]
  found: dict[str, Path] = {}

  for base in candidates:
    if not base.exists():
      continue
    for name in target_names:
      if name in found:
        continue
      exact_path = base / name
      lower_path = base / name.lower()
      if exact_path.is_dir():
        found[name] = exact_path
      elif lower_path.is_dir():
        found[name] = lower_path
  return found


@st.cache_data
def load_exam_chunks(exam_dirs: dict[str, Path], chunk_size: int = 1200) -> list[dict]:
    chunks = []
    for exam_label, folder in exam_dirs.items():
        for pdf_path in sorted(folder.rglob("*.pdf")):
            try:
                reader = PdfReader(pdf_path)
            except Exception:
                continue
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                for i in range(0, len(text), chunk_size):
                    chunks.append({
                        "exam": exam_label.title(),
                        "source": pdf_path.name,
                        "page": page_num,
                        "text": text[i : i + chunk_size],
                    })
    return chunks


def retrieve_context(query: str, chunks: list[dict], top_k: int = 3) -> str:
    if not chunks:
        return "No PDF text available."
    terms = {w.lower().strip(".,?!:;()[]") for w in query.split() if len(w) > 3}
    if not terms:
        return "\n\n---\n\n".join(
            f"[{c['exam']} | {c['source']} p.{c['page']}]\n{c['text']}"
            for c in chunks[:top_k]
        )
    scored = []
    for c in chunks:
        low = c["text"].lower()
        score = sum(1 for t in terms if t in low)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for score, c in scored[:top_k] if score > 0] or chunks[:top_k]
    return "\n\n---\n\n".join(
        f"[{c['exam']} | {c['source']} p.{c['page']}]\n{c['text']}" for c in selected
    )


def save_log(entry: dict) -> None:
    logs = []
    if LOG_FILE.exists():
        try:
            logs = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logs = []
    logs.append(entry)
    LOG_FILE.write_text(json.dumps(logs, indent=2), encoding="utf-8")


# --- Sidebar Controls ---
st.title("Econometrics Socratic Tutor")
st.caption(
    "Grounded in your Exam 1, Exam 2, and Exam 3 materials • Guides reasoning"
    " before revealing full answers"
)

api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input(
    "Gemini API Key", type="password"
)

exam_dirs = find_exam_dirs()
all_chunks = load_exam_chunks(exam_dirs)

exam_options = ["All Exams"] + [label.title() for label in exam_dirs.keys()]
selected_exam = st.sidebar.selectbox("Focus Scope", exam_options)

filtered_chunks = (
    all_chunks
    if selected_exam == "All Exams"
    else [c for c in all_chunks if c["exam"] == selected_exam]
)

st.sidebar.markdown(f"**Detected folders:** {', '.join(exam_dirs.keys()) or 'None'}")
st.sidebar.markdown(f"**Indexed PDF chunks:** {len(filtered_chunks)}")

# --- Chat Loop ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input(
    "Paste an econometrics question, concept, or your attempt..."
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        reply = (
            "Enter your Gemini API key in the sidebar (or set `GEMINI_API_KEY`) to"
            " start a tutoring turn."
        )
    else:
        client = genai.Client(api_key=api_key)
        context = retrieve_context(prompt, filtered_chunks)
        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages[-6:]
        )

        system_instruction = """You are a Socratic econometrics tutor grounded in the student's exam and course PDFs.
- Do NOT hand over the complete solution or final answer immediately on a new question.
- Start with 1-2 targeted diagnostic questions about intuition, key assumptions, definitions, or what the student already sees in the problem.
- Ground hints in the provided course excerpts when relevant, and explicitly note if something isn't covered in the excerpts.
- Once the student makes a genuine attempt or asks for a hint after trying, give a concise next step and another check rather than dumping everything at once."""

        full_prompt = f"""COURSE EXCERPTS ({selected_exam}):
{context}

RECENT CONVERSATION:
{history}

Student's latest message: {prompt}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
        reply = response.text or "No response generated."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

    save_log({
        "timestamp": datetime.now().isoformat(),
        "scope": selected_exam,
        "user_message": prompt,
        "tutor_reply": reply,
    })
