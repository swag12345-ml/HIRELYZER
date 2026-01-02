def evaluate_interview_answer(answer: str, question: str = None):
    """
    Uses an LLM to strictly evaluate an interview answer.
    Returns (score out of 5, feedback string).
    """
    from llm_manager import call_llm
    import re
    import streamlit as st

    # Empty check
    if not answer.strip():
        return 0, "⚠️ No answer provided."

    # 🔹 LLM Prompt (STRICTER)
    prompt = f"""
    You are an expert technical interview evaluator.

    ### Task:
    Evaluate the candidate's answer to the question below.
    Be STRICT. Only give high scores if the answer is technically correct, relevant, and detailed.

    ### Question:
    {question if question else "N/A"}

    ### Candidate Answer:
    {answer}

    ### Strict Scoring Rubric:
    - 5 = Exceptional: Fully correct, highly relevant, clear, detailed, technically accurate.
    - 4 = Good: Mostly correct and relevant, but missing some depth/clarity.
    - 3 = Average: Partially correct OR generic, but somewhat relevant.
    - 2 = Weak: Mostly irrelevant, shallow, or major gaps in correctness.
    - 1 = Poor: Completely irrelevant, incoherent, or very wrong.
    - 0 = No answer / total nonsense.

    ### Output Format:
    Score: <number between 0 and 5>
    Feedback: <constructive feedback in 1–2 sentences>
    """

    try:
        # Call LLM
        response = call_llm(prompt, session=st.session_state).strip()

        # Extract Score
        score_match = re.search(r"Score:\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 1  # stricter fallback

        # Extract Feedback
        feedback_match = re.search(r"Feedback:\s*(.+)", response)
        feedback = feedback_match.group(1).strip() if feedback_match else "Answer was unclear or irrelevant."

        # ✅ Keep score in 0–5 range
        score = max(0, min(score, 5))

    except Exception as e:
        score = 1
        feedback = f"⚠️ Evaluation fallback due to error: {e}"

    return score, feedback


def evaluate_interview_answer_for_scores(answer: str, question: str, difficulty: str, role: str = "", domain: str = ""):
    """
    UPGRADED: Intelligent evaluation with chain-of-thought reasoning and structured feedback.
    Uses JSON-based parsing for robustness and provides detailed, actionable feedback.

    Returns dict with keys: knowledge, communication, relevance, feedback (list), followup

    Features:
    - Chain-of-thought evaluation: extracts key concepts, identifies strengths/gaps
    - Structured feedback: detailed paragraph with specific, actionable insights
    - Difficulty calibration: Easy (encouraging), Medium (balanced), Hard (strict)
    - JSON-based parsing for reliability
    """
    from llm_manager import call_llm
    import json
    import streamlit as st

    # Empty check or junk answers
    if not answer.strip() or answer == "⚠️ No Answer" or len(answer.strip()) < 3:
        return {
            "knowledge": 0,
            "communication": 0,
            "relevance": 0,
            "feedback": "No answer provided. Try using the STAR method: Situation, Task, Action, Result. Provide specific examples from your experience to demonstrate your understanding and capabilities.",
            "followup": ""
        }

    # Check for obvious junk answers (single character, just symbols, etc.)
    if len(answer.strip()) == 1 or not any(c.isalnum() for c in answer):
        return {
            "knowledge": 0,
            "communication": 0,
            "relevance": 0,
            "feedback": "Answer appears incomplete or invalid. Please provide a meaningful response with technical details and structure your answer clearly with concrete examples from your experience.",
            "followup": ""
        }

    # STRICTER JUNK FILTERING: Check word count and meaningful tokens
    words = answer.strip().split()
    meaningful_words = [w for w in words if len(w) > 2 and any(c.isalpha() for c in w)]

    if len(words) < 5 or len(meaningful_words) < 2:
        return {
            "knowledge": 0,
            "communication": 0,
            "relevance": 0,
            "feedback": "Answer too short or lacks substance. Provide a detailed response with at least 3-4 sentences and include specific examples or technical details to demonstrate your understanding.",
            "followup": ""
        }

    # Difficulty-based evaluation guidance
    difficulty_guidance = {
        "Easy": {
            "tone": "encouraging and forgiving",
            "expectations": "basic understanding and general concepts",
            "scoring": "Give partial credit for effort. Score 5-10 for reasonable attempts, 3-4 for weak but present answers, 0-2 for irrelevant/junk.",
            "feedback_style": "positive and encouraging with gentle improvement tips"
        },
        "Medium": {
            "tone": "balanced and realistic",
            "expectations": "scenario-based thinking, some technical depth, and practical examples",
            "scoring": "Score 6-10 for good answers, 3-5 for incomplete/basic answers, 0-2 for poor/irrelevant.",
            "feedback_style": "constructive and specific with clear improvement areas"
        },
        "Hard": {
            "tone": "strict and technical",
            "expectations": "deep technical knowledge, system design thinking, edge cases, and comprehensive understanding",
            "scoring": "Score 7-10 for excellent answers, 4-6 for adequate but incomplete, 0-3 for weak/incorrect.",
            "feedback_style": "concise and technical with precise critique"
        }
    }

    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["Medium"])

    # Build context for relevance checking
    context_info = f" for {role} in {domain}" if role and domain else ""

    # UPGRADED CHAIN-OF-THOUGHT EVALUATION PROMPT
    prompt = f"""You are an expert technical interviewer evaluating a candidate's answer{context_info}.

QUESTION: {question}
CANDIDATE'S ANSWER: {answer}
DIFFICULTY LEVEL: {difficulty}

EVALUATION APPROACH ({guidance['tone']}):
Expected: {guidance['expectations']}
Scoring: {guidance['scoring']}
Feedback Style: {guidance['feedback_style']}

STEP-BY-STEP EVALUATION PROCESS:

STEP 1 - EXTRACT KEY CONCEPTS FROM THE QUESTION:
List 3-5 technical concepts, keywords, or expected topics the question is asking about.

STEP 2 - ANALYZE THE ANSWER:
✅ STRENGTHS: What did the candidate do well? Which concepts did they cover? What was clear or correct?
⚠️ GAPS/IMPROVEMENTS: What's missing? What's incorrect? What could be clearer?

STEP 3 - SCORE THE ANSWER (1-10 scale):
- Knowledge: Technical correctness, depth, and completeness (did they cover key concepts?)
- Communication: Clarity, structure, and articulation (was it easy to follow?)
- Relevance: How directly does this answer the question? Is it on-topic?

STEP 4 - GENERATE DETAILED FEEDBACK (2-4 comprehensive paragraphs):
Provide detailed, flowing feedback that covers:
- Specific strengths: What they did well, which concepts they covered correctly, and what was clear
- Specific gaps or areas for improvement: What key concepts or details they missed, what could be more accurate
- Actionable recommendations: Concrete suggestions for improvement with examples
- Overall assessment: A brief summary of their understanding level

Write feedback as natural, flowing paragraphs (not bullet points). Make it detailed, specific to their answer, and constructive.

{"STEP 5 - FOLLOW-UP QUESTION: Generate ONE probing follow-up question that digs deeper based on their answer." if difficulty == "Hard" else ""}

OUTPUT FORMAT (strict JSON):
{{
  "key_concepts": ["concept1", "concept2", "concept3"],
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "knowledge": <number 1-10>,
  "communication": <number 1-10>,
  "relevance": <number 1-10>,
  "feedback": "Detailed, comprehensive feedback in 2-4 flowing paragraphs. Be specific about what the candidate did well, what they missed, and how they can improve. Reference actual content from their answer. Make it constructive, actionable, and personalized."{',\n  "followup": "One probing follow-up question"' if difficulty == "Hard" else ''}
}}

IMPORTANT RULES:
- If answer is off-topic or from wrong domain, set relevance to 0-2
- If answer is junk/minimal, set all scores to 0-2
- Feedback must be specific to THIS answer, not generic templates
- Reference actual content from the candidate's answer in feedback
- Each feedback point should feel personalized and human

Provide ONLY the JSON output, no additional text."""

    try:
        response = call_llm(prompt, session=st.session_state).strip()

        # Clean response - remove markdown code blocks if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

        # Parse JSON response
        result = json.loads(response)

        # Extract and validate scores
        knowledge = int(result.get("knowledge", 1))
        communication = int(result.get("communication", 1))
        relevance = int(result.get("relevance", 1))

        # Clamp scores to 0-10 range
        knowledge = max(0, min(10, knowledge))
        communication = max(0, min(10, communication))
        relevance = max(0, min(10, relevance))

        # Extract feedback (should be a detailed string, not a list)
        feedback = result.get("feedback", "")

        # If feedback comes as a list (fallback), join it into paragraphs
        if isinstance(feedback, list):
            feedback = "\n\n".join(feedback)

        # Ensure we have substantial feedback
        if not feedback or len(feedback.strip()) < 50:
            feedback = "Your answer shows some understanding, but could benefit from more technical depth and specific examples. Consider structuring your response more clearly and providing concrete details from your experience. Focus on addressing all aspects of the question comprehensively."

        # Extract follow-up question
        followup = result.get("followup", "") if difficulty == "Hard" else ""

        return {
            "knowledge": knowledge,
            "communication": communication,
            "relevance": relevance,
            "feedback": feedback,  # Now a string, not a list
            "followup": followup
        }

    except json.JSONDecodeError as e:
        # Fallback: try to extract scores from non-JSON response
        import re
        try:
            knowledge = int(re.search(r'"?knowledge"?\s*:\s*(\d+)', response, re.IGNORECASE).group(1))
            communication = int(re.search(r'"?communication"?\s*:\s*(\d+)', response, re.IGNORECASE).group(1))
            relevance = int(re.search(r'"?relevance"?\s*:\s*(\d+)', response, re.IGNORECASE).group(1))

            # Extract feedback (try both string and array format)
            feedback_match = re.search(r'"feedback"\s*:\s*"([^"]+)"', response, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1)
            else:
                # Fallback: try array format and join
                feedback_array_match = re.search(r'"feedback"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                if feedback_array_match:
                    feedback_text = feedback_array_match.group(1)
                    feedback_items = [f.strip(' "\'') for f in re.findall(r'"([^"]+)"', feedback_text)]
                    feedback = "\n\n".join(feedback_items) if feedback_items else "Answer evaluated but formatting unclear. Provide more structured responses with clear examples and explanations."
                else:
                    feedback = "Answer evaluated but formatting unclear. Provide more structured responses with clear examples and explanations."

            return {
                "knowledge": max(0, min(10, knowledge)),
                "communication": max(0, min(10, communication)),
                "relevance": max(0, min(10, relevance)),
                "feedback": feedback if isinstance(feedback, str) else "\n\n".join(feedback[:5]),
                "followup": ""
            }
        except:
            pass

    except Exception as e:
        pass

    # Final fallback based on difficulty
    fallback_scores = {"Easy": 3, "Medium": 2, "Hard": 1}
    fallback_score = fallback_scores.get(difficulty, 2)

    return {
        "knowledge": fallback_score,
        "communication": fallback_score,
        "relevance": fallback_score,
        "feedback": "Unable to evaluate properly. Please provide a clear, structured answer. Use the STAR method for behavioral questions and include technical details and examples for technical questions.",
        "followup": ""
    }


def get_ist_time():
    """Get current time in IST timezone"""
    try:
        from datetime import datetime
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
    except:
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_user_action(username: str, action: str):
    """Log user actions - placeholder for compatibility"""
    pass


def create_interview_database():
    """Create interview_results table if not exists"""
    import sqlite3
    try:
        conn = sqlite3.connect('resume_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interview_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT,
                domain TEXT,
                avg_score REAL,
                total_questions INTEGER,
                completed_on TEXT,
                feedback_summary TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        import streamlit as st
        st.error(f"Database error: {e}")


def save_interview_result(username: str, role: str, domain: str, avg_score: float, total_questions: int, feedback_summary: str):
    """Save interview result to database"""
    import sqlite3
    try:
        conn = sqlite3.connect('resume_data.db')
        cursor = conn.cursor()
        completed_on = get_ist_time()
        cursor.execute("""
            INSERT INTO interview_results (username, role, domain, avg_score, total_questions, completed_on, feedback_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, role, domain, avg_score, total_questions, completed_on, feedback_summary))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        import streamlit as st
        st.error(f"Failed to save interview result: {e}")
        return False


def format_feedback_text(feedback):
    """
    Format feedback text into bullet points for clean display
    """
    import re
    import html
    sentences = re.split(r'(?<=\.)\s+', feedback.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    formatted = "<b>💡 Improvement Tips:</b><br><ul style='margin-top:5px;'>"
    for s in sentences:
        # Escape HTML special characters to display tags like <header>, <section>, etc.
        safe_sentence = html.escape(s)
        formatted += f"<li>{safe_sentence}</li>"
    formatted += "</ul>"
    return formatted


def generate_interview_pdf_report(username, role, domain, completed_on, questions, answers, scores, feedbacks, overall_avg, badge, difficulty="Medium"):
    """
    Generate PDF report for interview using xhtml2pdf

    FIXED: Now shows full answers (up to 2000 chars) instead of truncating at 500
    FIXED: Added follow-up questions for Hard difficulty interviews
    """
    try:
        from xhtml2pdf import pisa
        from io import BytesIO

        # Build XHTML content
        xhtml = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #00c3ff; text-align: center; }}
                h2 {{ color: #0099cc; margin-top: 20px; }}
                .header {{ background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .question-block {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; page-break-inside: avoid; }}
                .score {{ font-weight: bold; color: #00c3ff; }}
                .feedback {{ color: #666; margin-top: 10px; padding: 10px; background: #f9f9f9; border-left: 3px solid #00c3ff; }}
                .feedback ul {{ margin: 5px 0 0 0; padding-left: 20px; }}
                .feedback li {{ margin: 8px 0; line-height: 1.5; }}
                .summary {{ background: #fffacd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .answer-text {{ white-space: pre-wrap; word-wrap: break-word; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Interview Practice Report</h1>
            <div class="header">
                <p><strong>Candidate:</strong> {username}</p>
                <p><strong>Role:</strong> {role}</p>
                <p><strong>Domain:</strong> {domain}</p>
                <p><strong>Date:</strong> {completed_on}</p>
            </div>
            <div class="summary">
                <h2>Overall Performance</h2>
                <p class="score">Average Score: {overall_avg:.1f}/10</p>
                <p><strong>Badge Earned:</strong> {badge}</p>
            </div>
            <h2>Detailed Q&A Review</h2>
        """

        # CRITICAL FIX: Add each question/answer/score/feedback with FULL answer (no truncation)
        for i, (q, a, score_dict, f) in enumerate(zip(questions, answers, scores, feedbacks), 1):
            # Ensure score_dict is a dictionary
            if isinstance(score_dict, dict):
                avg_q_score = (score_dict.get('knowledge', 5) + score_dict.get('communication', 5) + score_dict.get('relevance', 5)) / 3
            else:
                # Fallback if score_dict is not a dict
                avg_q_score = 5.0
                score_dict = {'knowledge': 5, 'communication': 5, 'relevance': 5}

            # Escape HTML special characters to prevent rendering issues
            q_escaped = q.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            a_escaped = a.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            # Handle feedback as string (convert list to paragraphs if needed)
            if isinstance(f, list):
                f_text = "\n\n".join(f)
            else:
                f_text = str(f)

            # Format feedback into bullet points
            import re
            sentences = re.split(r'(?<=\.)\s+', f_text.strip())
            sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
            bullet_feedback = "<b>💡 Improvement Tips:</b><ul>"
            for sent in sentences:
                sent_escaped = sent.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                bullet_feedback += f"<li>{sent_escaped}</li>"
            bullet_feedback += "</ul>"
            f_escaped = bullet_feedback

            # SHOW FULL ANSWER - NO TRUNCATION IN PDF
            answer_display = a_escaped

            # Get follow-up question if exists (for Hard difficulty)
            followup_text = ""
            if difficulty == "Hard" and isinstance(score_dict, dict) and score_dict.get('followup'):
                followup_escaped = score_dict['followup'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                followup_text = f"""<div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-radius: 5px;">
                    <strong>Follow-up Question (for Hard interviews):</strong><br/>
                    {followup_escaped}
                </div>"""

            xhtml += f"""
            <div class="question-block">
                <h3>Question {i}</h3>
                <p><strong>Q:</strong> {q_escaped}</p>
                <div class="answer-text"><strong>Your Answer:</strong><br/>{answer_display}</div>
                <p class="score">Knowledge: {score_dict.get('knowledge', 0)}/10 | Communication: {score_dict.get('communication', 0)}/10 | Relevance: {score_dict.get('relevance', 0)}/10</p>
                <p class="score">Question Score: {avg_q_score:.1f}/10</p>
                <div class="feedback">{f_escaped}</div>
                {followup_text}
            </div>
            """

        xhtml += """
        </body>
        </html>
        """

        # Convert to PDF
        pdf_out = BytesIO()
        pisa_status = pisa.CreatePDF(xhtml, dest=pdf_out)
        pdf_out.seek(0)

        if pisa_status.err:
            return None

        return pdf_out.getvalue()

    except Exception as e:
        import streamlit as st
        st.error(f"PDF generation failed: {e}")
        return None


import streamlit as st
import plotly.graph_objects as go
from courses import COURSES_BY_CATEGORY, RESUME_VIDEOS, INTERVIEW_VIDEOS, get_courses_for_role
from llm_manager import call_llm
import time
import threading
import json


import json
import time
import re
import streamlit as st

# ======================================================
# RESUME TEXT EXTRACTION (pdfplumber + OCR fallback)
# ======================================================
def extract_resume_text_from_pdf(pdf_file):
    """
    Robust resume extraction:
    - pdfplumber for text-based & two-column resumes
    - OCR fallback for scanned/image resumes
    """

    text = ""

    # ---------- PRIMARY: pdfplumber ----------
    try:
        import pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = text.strip()
    except Exception:
        text = ""

    # ---------- FALLBACK: OCR ----------
    if len(text.split()) < 120:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(pdf_file.getvalue())
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img)

            if len(ocr_text.split()) > len(text.split()):
                text = ocr_text.strip()
        except Exception:
            st.warning("OCR fallback failed. Resume may be image-heavy.")

    # ---------- FINAL VALIDATION ----------
    if not text or len(text.split()) < 80:
        st.warning("Resume text extraction was weak. Some questions may be generic.")
        return None

    return text


# ======================================================
# RESUME ANALYSIS USING LLM (IMPROVED PROMPT)
# ======================================================
def analyze_resume_with_llm(resume_text):
    """
    Analyze resume using LLM to extract INTERVIEW-RELEVANT structured information
    """

    prompt = f"""
You are a senior technical interviewer and resume screening expert.

Analyze the resume below and extract ONLY the most interview-relevant information.
Focus on technical depth, real-world work, and ownership.
IGNORE generic soft skills unless strongly implied by technical work.

RESUME TEXT:
{resume_text}

Return ONLY a valid JSON object with this exact structure:

{{
  "skills": [
    "Core technical skill clearly demonstrated in projects or experience"
  ],
  "projects": [
    "Project name – what was built, tech used, and key technical challenge solved"
  ],
  "experience": [
    "Role at company – main technical responsibility and impact"
  ],
  "technologies": [
    "Primary technologies actually used (not buzzwords)"
  ]
}}

STRICT RULES:
- Prefer HARD technical skills over soft skills
- Extract ONLY skills clearly demonstrated
- Rank items by importance (most interview-worthy first)
- Avoid generic terms like 'problem solving', 'communication'
- Projects MUST mention tech used
- Experience MUST show ownership or responsibility
- Extract:
  - 4–6 skills
  - 2–4 projects
  - 2–4 experience entries
  - 4–6 technologies
- Keep entries concise but specific
- Output ONLY JSON (no markdown, no explanations)

JSON:
"""

    try:
        response = call_llm(prompt, session=st.session_state).strip()

        # Clean markdown if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.lower().startswith("json"):
                response = response[4:]
            response = response.strip()

        resume_data = json.loads(response)

        return {
            "skills": resume_data.get("skills", [])[:6],
            "projects": resume_data.get("projects", [])[:4],
            "experience": resume_data.get("experience", [])[:4],
            "technologies": resume_data.get("technologies", [])[:6]
        }

    except Exception:
        st.warning("Resume analysis failed. Using fallback data.")
        return {
            "skills": ["Basic Programming Knowledge"],
            "projects": ["Personal Technical Project"],
            "experience": ["General Technical Experience"],
            "technologies": ["General Tech Stack"]
        }


# ======================================================
# RESUME-BASED QUESTION GENERATION
# ======================================================
def generate_resume_based_questions(resume_context, role, domain, difficulty, num_questions=3):
    """
    Generate interview questions strictly based on resume context
    """

    skills = resume_context.get("skills", [])
    projects = resume_context.get("projects", [])
    experience = resume_context.get("experience", [])
    technologies = resume_context.get("technologies", [])

    prompt = f"""
You are a technical interviewer.

Generate EXACTLY {num_questions} interview questions based ONLY on the candidate's resume.

RESUME CONTEXT:
- Skills: {', '.join(skills[:4])}
- Projects: {', '.join(projects[:2])}
- Experience: {', '.join(experience[:2])}
- Technologies: {', '.join(technologies[:4])}

Target Role: {role}
Domain: {domain}
Difficulty: {difficulty}

RULES:
- Every question MUST reference resume content
- Ask like a real interviewer
- Difficulty:
  - Easy: explanation & fundamentals
  - Medium: scenarios & decisions
  - Hard: deep technical trade-offs or design
- Output ONLY questions
- One question per line
- No numbering, no prefixes

Generate now:
"""

    try:
        response = call_llm(prompt, session=st.session_state)
        raw_questions = [q.strip() for q in response.split("\n") if q.strip()]

        cleaned_questions = []
        for q in raw_questions:
            q = re.sub(r'^[\d\)\.\-•\*]+\s*', '', q).strip()
            if len(q) > 15:
                cleaned_questions.append(q)
            if len(cleaned_questions) >= num_questions:
                break

        while len(cleaned_questions) < num_questions:
            cleaned_questions.append(
                f"Explain your most significant project and the technical decisions you made."
            )

        return cleaned_questions[:num_questions]

    except Exception:
        return [
            "Walk us through your most technically challenging project.",
            "What design or implementation decisions did you personally make?",
            "How does your experience prepare you for this role?"
        ]


# ======================================================
# RESUME SCANNING ANIMATION
# ======================================================
def show_resume_scanning_animation():
    """Animated resume scanning UI"""

    status = st.empty()
    progress = st.empty()

    steps = [
        ("📄 Reading resume...", 0.2),
        ("🔍 Extracting key skills...", 0.4),
        ("📊 Evaluating experience...", 0.6),
        ("🧠 Understanding projects...", 0.8),
        ("🎯 Preparing interview questions...", 1.0),
    ]

    for text, value in steps:
        status.markdown(
            f"<h4 style='text-align:center;color:#00c3ff'>{text}</h4>",
            unsafe_allow_html=True
        )
        progress.progress(value)
        time.sleep(0.6)

    status.empty()
    progress.empty()


with tab4:
    # Inject CSS styles (keeping existing styles)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .header-box {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 25%, #2d3561 50%, #3f4787 75%, #5158ae 100%);
            border: 2px solid transparent;
            background-clip: padding-box;
            position: relative;
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 35px;
            box-shadow: 
                0 8px 32px rgba(0, 195, 255, 0.15),
                0 4px 16px rgba(0, 195, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            overflow: hidden;
        }

        .header-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, #00c3ff, #0066cc, #00c3ff, #0066cc);
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
            z-index: -1;
            border-radius: 20px;
            padding: 2px;
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask-composite: exclude;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .header-box h2 {
            font-size: 32px;
            color: #ffffff;
            margin: 0;
            font-weight: 700;
            text-shadow: 
                0 0 20px rgba(0, 195, 255, 0.5),
                0 2px 4px rgba(0, 0, 0, 0.3);
            letter-spacing: -0.5px;
        }

        .glow-header {
            font-size: 24px;
            text-align: center;
            color: #00c3ff;
            text-shadow: 
                0 0 20px rgba(0, 195, 255, 0.8),
                0 0 40px rgba(0, 195, 255, 0.4);
            margin: 20px 0 15px 0;
            font-weight: 600;
            letter-spacing: -0.3px;
            animation: pulse 3s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.9; transform: scale(1.02); }
        }

        .stRadio > div {
            flex-direction: row !important;
            justify-content: center !important;
            gap: 16px;
            flex-wrap: wrap;
        }

        .stRadio label {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00c3ff;
            color: #00c3ff;
            padding: 14px 24px;
            margin: 6px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
            min-width: 190px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 4px 15px rgba(0, 195, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .stRadio label::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 195, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .stRadio label:hover {
            background: linear-gradient(135deg, #00c3ff15 0%, #00c3ff25 100%);
            transform: translateY(-2px);
            box-shadow: 
                0 8px 25px rgba(0, 195, 255, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }

        .stRadio label:hover::before {
            left: 100%;
        }

        .stRadio input:checked + div > label {
            background: linear-gradient(135deg, #00c3ff 0%, #0099cc 100%);
            color: #000000;
            font-weight: 600;
            transform: scale(1.05);
            box-shadow: 
                0 8px 30px rgba(0, 195, 255, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }

        .card {
            background: linear-gradient(135deg, #0f1419 0%, #1a2332 25%, #253447 50%, #30455c 75%, #3b5671 100%);
            border: 2px solid transparent;
            border-radius: 16px;
            padding: 20px 25px;
            margin: 12px 0;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 4px 20px rgba(0, 195, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, #00c3ff, #0066cc);
            z-index: -1;
            border-radius: 16px;
            padding: 2px;
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask-composite: exclude;
            opacity: 0.8;
        }

        .card::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.6s;
        }

        .card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 
                0 12px 40px rgba(0, 195, 255, 0.25),
                0 8px 20px rgba(0, 195, 255, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .card:hover::after {
            left: 100%;
        }

        .card a {
            color: #00c3ff;
            font-weight: 600;
            font-size: 16px;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            text-shadow: 0 0 10px rgba(0, 195, 255, 0.3);
        }

        .card a:hover {
            color: #ffffff;
            text-decoration: none;
            text-shadow: 
                0 0 15px rgba(255, 255, 255, 0.5),
                0 0 30px rgba(0, 195, 255, 0.3);
            transform: translateX(4px);
        }

        /* Enhanced selectbox styling */
        .stSelectbox > div > div {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00c3ff;
            border-radius: 10px;
            color: #00c3ff;
        }

        .stSelectbox > div > div:hover {
            box-shadow: 0 0 15px rgba(0, 195, 255, 0.3);
        }

        /* Enhanced subheader styling */
        .stApp h3 {
            color: #00c3ff;
            text-shadow: 0 0 10px rgba(0, 195, 255, 0.5);
            font-weight: 600;
            margin-bottom: 20px;
        }

        /* Learning path container */
        .learning-path-container {
            text-align: center;
            margin: 30px 0 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, rgba(0, 195, 255, 0.05) 0%, rgba(0, 195, 255, 0.1) 100%);
            border-radius: 12px;
            border: 1px solid rgba(0, 195, 255, 0.2);
        }

        .learning-path-text {
            color: #00c3ff;
            font-weight: 600;
            font-size: 20px;
            text-shadow: 0 0 15px rgba(0, 195, 255, 0.6);
            letter-spacing: -0.3px;
        }

        /* Video container enhancements */
        .stVideo {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }

        .stVideo:hover {
            transform: scale(1.02);
        }

        /* Info message styling */
        .stAlert {
            background: linear-gradient(135deg, rgba(0, 195, 255, 0.1) 0%, rgba(0, 195, 255, 0.05) 100%);
            border: 1px solid rgba(0, 195, 255, 0.3);
            border-radius: 10px;
        }

        /* New styles for quiz and interview sections */
        .quiz-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00c3ff;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 20px rgba(0, 195, 255, 0.15);
        }

        .badge-container {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, rgba(0, 195, 255, 0.12) 0%, rgba(0, 195, 255, 0.06) 100%);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(0, 195, 255, 0.25);
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .score-display {
            font-size: 64px;
            font-weight: bold;
            color: #00d4ff;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
            letter-spacing: 2px;
        }

        .role-selector {
            background: linear-gradient(135deg, rgba(0, 195, 255, 0.05) 0%, rgba(0, 195, 255, 0.1) 100%);
            border: 1px solid rgba(0, 195, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
        }

        /* Course tile styling */
        .course-tile {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00c3ff;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .course-tile:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 195, 255, 0.3);
        }

        .course-title {
            color: #00c3ff;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .course-description {
            color: #ffffff;
            font-size: 14px;
            margin-bottom: 15px;
            line-height: 1.4;
        }

        .difficulty-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 15px;
        }

        .difficulty-beginner {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }

        .difficulty-intermediate {
            background: linear-gradient(135deg, #FF9800, #f57c00);
            color: white;
        }

        .difficulty-advanced {
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
        }

        .course-link-btn {
            background: linear-gradient(135deg, #00c3ff, #0099cc);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            display: inline-block;
            transition: all 0.3s ease;
        }

        .course-link-btn:hover {
            background: linear-gradient(135deg, #0099cc, #007acc);
            transform: translateX(2px);
            text-decoration: none;
            color: white;
        }

        /* Radar chart container */
        .radar-container {
            background: linear-gradient(135deg, rgba(0, 195, 255, 0.05) 0%, rgba(0, 195, 255, 0.1) 100%);
            border: 1px solid rgba(0, 195, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }

        /* Timer styling */
        .timer-container {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 12px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
        }

        .timer-display {
            font-size: 24px;
            font-weight: bold;
            color: #ffd700;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        }

        .timer-urgent {
            color: #ff4444;
            text-shadow: 0 0 15px rgba(255, 68, 68, 0.8);
            animation: pulse 1s ease-in-out infinite;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header (keeping existing)
    st.markdown("""
        <div class="header-box">
            <h2>📚 Recommended Learning Hub</h2>
        </div>
    """, unsafe_allow_html=True)

    # Subheader (keeping existing)
    st.markdown('<div class="glow-header">🎓 Explore Career Resources</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc; font-size: 16px; margin-bottom: 25px;'>Curated courses and videos for your career growth, resume tips, and interview success.</p>", unsafe_allow_html=True)

    # Learning path label (keeping existing)
    st.markdown("""
        <div class="learning-path-container">
            <span class="learning-path-text">
                🧭 Choose Your Learning Path
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Updated Radio buttons with new options
    st.markdown("""
        <div style="display: flex; justify-content: center; width: 100%;">
            <div style="display: flex; justify-content: center; gap: 16px;">
    """, unsafe_allow_html=True)

    # Check if page changed away from AI Interview Coach - stop interview if so
    previous_page = st.session_state.get('previous_page_selection', None)

    page = st.radio(
        label="Select Learning Option",
        options=["Courses by Role", "Resume Videos", "Interview Videos",  "AI Interview Coach 🤖"],
        horizontal=True,
        key="page_selection",
        label_visibility="collapsed"
    )

    # STOP INTERVIEW ON TAB CHANGE
    if previous_page == "AI Interview Coach 🤖" and page != "AI Interview Coach 🤖":
        # User switched away from AI Interview Coach - reset interview state
        if st.session_state.get('dynamic_interview_started', False) and not st.session_state.get('dynamic_interview_completed', False):
            st.session_state.dynamic_interview_started = False
            st.session_state.dynamic_interview_completed = True

    # Update previous page for next comparison
    st.session_state.previous_page_selection = page

    st.markdown("</div></div>", unsafe_allow_html=True)

    # NEW: Index-based difficulty function (replaces keyword-based)
    def get_course_difficulty_by_index(index):
        if index == 0:
            return "Beginner"
        elif index in [1, 2]:
            return "Intermediate"
        else:
            return "Advanced"

    # Helper functions for dynamic question generation
    def generate_career_quiz_questions(domain, role):
        """Generate role-specific career quiz questions"""
        questions = []
        
        # Role-specific question templates
        role_templates = {
            "Software Development and Engineering": {
                "Frontend Developer": [
                    {
                        "question": "Which aspect of web development excites you most?",
                        "options": [
                            "Creating beautiful, interactive user interfaces",
                            "Building responsive designs that work on all devices", 
                            "Optimizing website performance and accessibility",
                            "Working with modern JavaScript frameworks"
                        ]
                    },
                    {
                        "question": "What's your preferred approach to styling?",
                        "options": [
                            "Writing custom CSS from scratch",
                            "Using CSS frameworks like Bootstrap or Tailwind",
                            "CSS-in-JS solutions for component-based styling", 
                            "CSS preprocessors like Sass or Less"
                        ]
                    },
                    {
                        "question": "Which tools do you enjoy working with most?",
                        "options": [
                            "React, Vue, or Angular for building SPAs",
                            "HTML5, CSS3, and vanilla JavaScript",
                            "Design tools like Figma or Adobe XD",
                            "Build tools like Webpack, Vite, or Parcel"
                        ]
                    }
                ],
                "Backend Developer": [
                    {
                        "question": "What backend architecture interests you most?",
                        "options": [
                            "RESTful API design and implementation",
                            "Microservices architecture and distributed systems",
                            "Database design and optimization",
                            "Server-side security and authentication"
                        ]
                    },
                    {
                        "question": "Which programming paradigm do you prefer?",
                        "options": [
                            "Object-oriented programming with Java/.NET",
                            "Functional programming with languages like Scala",
                            "Dynamic languages like Python or JavaScript",
                            "Systems programming with Go or Rust"
                        ]
                    },
                    {
                        "question": "What type of backend challenges excite you?",
                        "options": [
                            "Scaling applications to handle millions of users",
                            "Integrating complex third-party services",
                            "Optimizing database queries and performance",
                            "Building robust error handling and monitoring"
                        ]
                    }
                ],
                "Full Stack Developer": [
                    {
                        "question": "What full-stack aspect appeals to you most?",
                        "options": [
                            "Building end-to-end features from UI to database",
                            "Managing the entire application development lifecycle",
                            "Working with both frontend and backend technologies",
                            "Understanding how all system components interact"
                        ]
                    },
                    {
                        "question": "Which tech stack interests you most?",
                        "options": [
                            "MERN (MongoDB, Express, React, Node.js)",
                            "MEAN (MongoDB, Express, Angular, Node.js)",
                            "Django + React/Vue for Python development",
                            "Ruby on Rails with modern frontend frameworks"
                        ]
                    }
                ],
                "Mobile App Developer": [
                    {
                        "question": "What type of mobile development interests you?",
                        "options": [
                            "Native iOS development with Swift",
                            "Native Android development with Kotlin/Java",
                            "Cross-platform development with React Native",
                            "Hybrid app development with Flutter"
                        ]
                    },
                    {
                        "question": "Which mobile development aspect excites you most?",
                        "options": [
                            "Creating intuitive mobile user experiences",
                            "Integrating with device hardware and sensors",
                            "Optimizing app performance and battery usage",
                            "Publishing apps to App Store and Google Play"
                        ]
                    }
                ],
                "Game Developer": [
                    {
                        "question": "What type of game development interests you?",
                        "options": [
                            "3D game development with Unity or Unreal Engine",
                            "2D indie game development and pixel art",
                            "Mobile gaming and casual game mechanics",
                            "VR/AR game development and immersive experiences"
                        ]
                    },
                    {
                        "question": "Which game development aspect excites you most?",
                        "options": [
                            "Game design and player experience",
                            "Graphics programming and visual effects",
                            "Game physics and realistic simulations",
                            "Multiplayer networking and real-time systems"
                        ]
                    }
                ]
            },
            "Data Science and Analytics": {
                "Data Scientist": [
                    {
                        "question": "Which data science task excites you most?",
                        "options": [
                            "Building predictive models and machine learning algorithms",
                            "Exploring large datasets to discover hidden patterns",
                            "Creating data visualizations and storytelling with data",
                            "Designing experiments and A/B testing strategies"
                        ]
                    },
                    {
                        "question": "What's your preferred approach to data analysis?",
                        "options": [
                            "Statistical modeling and hypothesis testing",
                            "Deep learning and neural networks",
                            "Feature engineering and data preprocessing",
                            "Time series analysis and forecasting"
                        ]
                    },
                    {
                        "question": "Which tools do you enjoy working with most?",
                        "options": [
                            "Python with pandas, scikit-learn, and TensorFlow",
                            "R for statistical computing and analysis",
                            "SQL for database querying and data manipulation",
                            "Jupyter notebooks for exploratory data analysis"
                        ]
                    }
                ],
                "Data Analyst": [
                    {
                        "question": "Which type of analysis interests you most?",
                        "options": [
                            "Business intelligence and performance dashboards",
                            "Customer behavior analysis and segmentation",
                            "Financial analysis and risk assessment",
                            "Market research and competitive analysis"
                        ]
                    },
                    {
                        "question": "What's your preferred way to present insights?",
                        "options": [
                            "Interactive dashboards with Tableau or Power BI",
                            "Statistical reports with clear recommendations",
                            "Data visualizations and infographics",
                            "Executive summaries and business presentations"
                        ]
                    }
                ],
                "Machine Learning Engineer": [
                    {
                        "question": "Which ML engineering task excites you most?",
                        "options": [
                            "Deploying models to production at scale",
                            "Building ML pipelines and automation systems",
                            "Optimizing model performance and efficiency",
                            "Implementing MLOps and model monitoring"
                        ]
                    },
                    {
                        "question": "What type of ML problems interest you?",
                        "options": [
                            "Computer vision and image processing",
                            "Natural language processing and text analysis",
                            "Recommendation systems and personalization",
                            "Reinforcement learning and autonomous systems"
                        ]
                    }
                ]
            },
            "Cloud Computing and DevOps": {
                "Cloud Architect": [
                    {
                        "question": "Which cloud architecture aspect interests you most?",
                        "options": [
                            "Designing scalable, fault-tolerant systems",
                            "Multi-cloud and hybrid cloud strategies",
                            "Cloud security and compliance frameworks",
                            "Cost optimization and resource management"
                        ]
                    },
                    {
                        "question": "What type of cloud solutions excite you?",
                        "options": [
                            "Serverless architectures and event-driven systems",
                            "Container orchestration with Kubernetes",
                            "Data lakes and analytics platforms",
                            "AI/ML platforms and managed services"
                        ]
                    }
                ],
                "DevOps Engineer": [
                    {
                        "question": "Which DevOps practice interests you most?",
                        "options": [
                            "Building CI/CD pipelines and automation",
                            "Infrastructure as Code with Terraform/CloudFormation",
                            "Container orchestration and microservices",
                            "Monitoring, logging, and observability"
                        ]
                    },
                    {
                        "question": "What type of automation excites you?",
                        "options": [
                            "Deployment automation and release management",
                            "Infrastructure provisioning and configuration",
                            "Testing automation and quality gates",
                            "Incident response and self-healing systems"
                        ]
                    }
                ],
                "Site Reliability Engineer": [
                    {
                        "question": "Which SRE responsibility interests you most?",
                        "options": [
                            "Maintaining system reliability and uptime",
                            "Performance optimization and capacity planning",
                            "Incident management and post-mortem analysis",
                            "Service level objectives and error budgets"
                        ]
                    },
                    {
                        "question": "What aspect of system reliability excites you?",
                        "options": [
                            "Building robust monitoring and alerting systems",
                            "Designing disaster recovery and backup strategies",
                            "Automating operational tasks and runbooks",
                            "Analyzing system performance and bottlenecks"
                        ]
                    }
                ]
            },
            "Cybersecurity": {
                "Security Analyst": [
                    {
                        "question": "Which security area interests you most?",
                        "options": [
                            "Threat detection and incident response",
                            "Vulnerability assessment and risk management",
                            "Security monitoring and SIEM analysis",
                            "Compliance and security policy development"
                        ]
                    },
                    {
                        "question": "What type of security challenges excite you?",
                        "options": [
                            "Investigating security breaches and forensics",
                            "Analyzing malware and attack patterns",
                            "Network security and firewall management",
                            "Identity and access management systems"
                        ]
                    }
                ],
                "Penetration Tester": [
                    {
                        "question": "Which penetration testing approach interests you?",
                        "options": [
                            "Web application security testing",
                            "Network penetration testing and infrastructure",
                            "Social engineering and phishing simulations",
                            "Mobile application security testing"
                        ]
                    },
                    {
                        "question": "What aspect of ethical hacking excites you?",
                        "options": [
                            "Finding vulnerabilities before malicious actors",
                            "Using creative techniques to bypass security",
                            "Helping organizations improve their defenses",
                            "Staying updated on latest attack methods"
                        ]
                    }
                ]
            },
            "UI/UX Design": {
                "UI Designer": [
                    {
                        "question": "Which UI design aspect interests you most?",
                        "options": [
                            "Creating visually stunning interface designs",
                            "Designing consistent design systems and components",
                            "Working with typography, colors, and visual hierarchy",
                            "Prototyping interactions and micro-animations"
                        ]
                    },
                    {
                        "question": "What type of design work excites you?",
                        "options": [
                            "Mobile app interface design",
                            "Web application and dashboard design",
                            "Icon design and visual asset creation",
                            "Brand identity and visual design systems"
                        ]
                    }
                ],
                "UX Designer": [
                    {
                        "question": "Which UX design activity interests you most?",
                        "options": [
                            "User research and persona development",
                            "Information architecture and user flows",
                            "Wireframing and prototype development",
                            "Usability testing and design validation"
                        ]
                    },
                    {
                        "question": "What aspect of user experience excites you?",
                        "options": [
                            "Solving complex user problems with simple solutions",
                            "Understanding user behavior and psychology",
                            "Designing accessible and inclusive experiences",
                            "Measuring and optimizing user engagement"
                        ]
                    }
                ]
            },
            "Project Management": {
                "Project Manager": [
                    {
                        "question": "Which project management aspect interests you most?",
                        "options": [
                            "Planning and scheduling project timelines",
                            "Coordinating teams and stakeholder communication",
                            "Risk management and problem-solving",
                            "Budget management and resource allocation"
                        ]
                    },
                    {
                        "question": "What type of projects excite you?",
                        "options": [
                            "Large-scale software development projects",
                            "Cross-functional digital transformation initiatives",
                            "Product launches and go-to-market strategies",
                            "Process improvement and organizational change"
                        ]
                    }
                ],
                "Product Manager": [
                    {
                        "question": "Which product management activity interests you most?",
                        "options": [
                            "Product strategy and roadmap development",
                            "User research and market analysis",
                            "Feature prioritization and requirement gathering",
                            "Go-to-market strategy and product launches"
                        ]
                    },
                    {
                        "question": "What aspect of product development excites you?",
                        "options": [
                            "Identifying user needs and pain points",
                            "Defining product vision and strategy",
                            "Working with engineering and design teams",
                            "Analyzing product metrics and user feedback"
                        ]
                    }
                ]
            }
        }

        # Get role-specific questions or generate generic ones
        if domain in role_templates and role in role_templates[domain]:
            questions = role_templates[domain][role]
        else:
            # Generate generic questions based on role name
            questions = [
                {
                    "question": f"How interested are you in pursuing a career as a {role}?",
                    "options": [
                        "Very interested - it's my dream job",
                        "Somewhat interested - I want to learn more",
                        "Moderately interested - it seems challenging",
                        "Not very interested - but I'm curious"
                    ]
                },
                {
                    "question": f"What attracts you most about the {role} role?",
                    "options": [
                        "The technical challenges and problem-solving",
                        "The creative aspects and innovation opportunities", 
                        "The career growth potential and salary",
                        "The impact on users and business outcomes"
                    ]
                }
            ]
        
        return questions

    # Helper function to generate fallback questions
    def self_generate_fallback_questions(role, domain, difficulty, count):
        """Generate fallback questions when LLM doesn't return enough"""
        if difficulty == "Easy":
            base_questions = [
                f"What interests you most about the {role} position?",
                f"Describe your basic understanding of {role} responsibilities.",
                f"What are the fundamental skills needed for {role}?",
                f"How do you stay updated with trends in {domain}?",
                f"Why do you want to work as a {role}?",
                f"What do you know about the {role} role?",
                f"Tell me about yourself and your interest in {role}.",
                f"What motivates you to pursue a career in {domain}?",
                f"Describe a project you've worked on related to {role}.",
                f"What are your career goals as a {role}?"
            ]
        elif difficulty == "Hard":
            base_questions = [
                f"Design a scalable system architecture for a {role} project handling millions of users.",
                f"Explain the trade-offs between different approaches in {domain} and when to use each.",
                f"How would you troubleshoot a critical production issue in a {role} context?",
                f"Describe your approach to mentoring junior team members as a {role}.",
                f"What are the biggest technical challenges facing {role} professionals today?",
                f"How would you architect a distributed system for {domain}?",
                f"Explain how you would optimize performance in a complex {role} project.",
                f"What advanced techniques do you use in {domain}?",
                f"Describe a time you made a critical technical decision as a {role}.",
                f"How do you approach system design for high availability in {domain}?"
            ]
        else:  # Medium
            base_questions = [
                f"Describe a challenging project you've worked on relevant to {role}.",
                f"How do you approach problem-solving in {domain}?",
                f"What tools and technologies are you most comfortable with for {role}?",
                f"Tell me about a time you had to learn a new skill for {role}.",
                f"How do you prioritize tasks when working as a {role}?",
                f"Describe your experience with {domain} technologies.",
                f"How do you handle tight deadlines as a {role}?",
                f"What's your approach to code quality in {domain}?",
                f"Tell me about a technical challenge you solved as a {role}.",
                f"How do you collaborate with team members in {domain}?"
            ]
        return base_questions[:count]

    # UPDATED: AI-Generated Questions using LLM with DIFFICULTY SUPPORT
    def generate_interview_questions_with_llm(domain, role, interview_type, num_questions, difficulty="Medium"):
        """
        Generate interview questions using LLM based on domain, role, type, and difficulty.

        FIXED: Now difficulty is passed into LLM prompt and affects question complexity
        """
        # Define difficulty-specific instructions
        difficulty_instructions = {
            "Easy": "Generate BASIC and INTRODUCTORY level questions. Focus on fundamental concepts, definitions, and simple scenarios. Questions should be suitable for entry-level candidates or those new to the field.",
            "Medium": "Generate SCENARIO-BASED and MODERATELY TECHNICAL questions. Include situational questions that require practical thinking and intermediate technical knowledge. Suitable for candidates with some experience.",
            "Hard": "Generate DEEP TECHNICAL, SYSTEM DESIGN, and COMPLEX PROBLEM-SOLVING questions. Include architecture decisions, trade-offs, scalability concerns, and advanced concepts. Suitable for senior-level candidates."
        }

        prompt = f"""You are an expert interviewer.

Generate EXACTLY {num_questions} unique {interview_type} interview questions
for the role of {role} in {domain}.

DIFFICULTY LEVEL: {difficulty}
{difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])}

CRITICAL REQUIREMENTS:
- Generate EXACTLY {num_questions} questions - no more, no less
- Keep each question concise (1-2 sentences max)
- Avoid duplicates
- Match the difficulty level specified above
- Output ONLY the questions, one per line
- DO NOT add numbering, bullet points, or any prefixes
- DO NOT add any introductory text or explanations

Output format example:
What is your experience with cloud technologies?
How would you handle a system outage?
Describe your approach to code reviews.

Generate exactly {num_questions} questions now:
"""

        try:
            response = call_llm(prompt, session=st.session_state)

            # Split by newlines and clean up
            raw_questions = [q.strip() for q in response.split('\n') if q.strip()]

            # Remove any numbering or bullet points more aggressively
            import re
            cleaned_questions = []
            for q in raw_questions:
                # Remove various prefixes: "1. ", "1) ", "- ", "• ", "* ", "Question 1:", etc.
                clean_q = re.sub(r'^[\d\)\.\-•\*]+\s*', '', q).strip()
                clean_q = re.sub(r'^Question\s*\d*\s*:?\s*', '', clean_q, flags=re.IGNORECASE).strip()

                # Only add if it's a meaningful question
                if clean_q and len(clean_q) > 15 and not clean_q.lower().startswith('generate') and not clean_q.lower().startswith('here'):
                    cleaned_questions.append(clean_q)

                # Stop if we have enough questions
                if len(cleaned_questions) >= num_questions:
                    break

            # If we got fewer questions than requested, try to pad with fallback
            if len(cleaned_questions) < num_questions:
                st.warning(f"Only generated {len(cleaned_questions)} questions, padding with fallback questions...")
                # Add fallback questions to meet the requirement
                fallback_needed = num_questions - len(cleaned_questions)
                fallback_qs = self_generate_fallback_questions(role, domain, difficulty, fallback_needed)
                cleaned_questions.extend(fallback_qs)

            # EXACT QUESTION COUNT: Enforce exact count
            cleaned_questions = cleaned_questions[:num_questions]
            return cleaned_questions

        except Exception as e:
            st.error(f"Failed to generate questions with LLM: {e}")
            # Fallback to static questions appropriate for difficulty
            if difficulty == "Easy":
                fallback_questions = [
                    f"What interests you most about the {role} position?",
                    f"Describe your basic understanding of {role} responsibilities.",
                    f"What are the fundamental skills needed for {role}?",
                    f"How do you stay updated with trends in {domain}?",
                    f"Why do you want to work as a {role}?"
                ]
            elif difficulty == "Hard":
                fallback_questions = [
                    f"Design a scalable system architecture for a {role} project handling millions of users.",
                    f"Explain the trade-offs between different approaches in {domain} and when to use each.",
                    f"How would you troubleshoot a critical production issue in a {role} context?",
                    f"Describe your approach to mentoring junior team members as a {role}.",
                    f"What are the biggest technical challenges facing {role} professionals today?"
                ]
            else:  # Medium
                fallback_questions = [
                    f"Describe a challenging project you've worked on relevant to {role}.",
                    f"How do you approach problem-solving in {domain}?",
                    f"What tools and technologies are you most comfortable with for {role}?",
                    f"Tell me about a time you had to learn a new skill for {role}.",
                    f"How do you prioritize tasks when working as a {role}?"
                ]
            return fallback_questions[:num_questions]

    # Badge system for gamification
    BADGE_CONFIG = {
        "career_quiz": {
            "novice": {"min_score": 0, "max_score": 40, "emoji": "🌱", "title": "Career Explorer"},
            "intermediate": {"min_score": 41, "max_score": 70, "emoji": "📚", "title": "Career Seeker"},
            "advanced": {"min_score": 71, "max_score": 100, "emoji": "🎯", "title": "Career Champion"}
        },
        "interview": {
            "needs_practice": {"min_score": 1.0, "max_score": 2.5, "emoji": "💪", "title": "Keep Practicing"},
            "good": {"min_score": 2.6, "max_score": 3.5, "emoji": "👍", "title": "Good Performer"},
            "excellent": {"min_score": 3.6, "max_score": 4.5, "emoji": "🌟", "title": "Star Performer"},
            "interview_ready": {"min_score": 4.6, "max_score": 5.0, "emoji": "🏆", "title": "Interview Ready"}
        }
    }

    def get_badge_for_score(score_type, score):
        """Get badge based on score type and value"""
        badges = BADGE_CONFIG.get(score_type, {})
        for badge_name, config in badges.items():
            if config["min_score"] <= score <= config["max_score"]:
                return config["emoji"], config["title"]
        return "🎖️", "Participant"

    def create_skill_radar_chart(skills_data):
        """Create a radar chart for skills using Plotly"""
        # Extract skills and values
        skills = list(skills_data.keys())
        values = list(skills_data.values())
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=skills,
            fill='toself',
            name='Skills',
            line=dict(color='#00c3ff', width=2),
            fillcolor='rgba(0, 195, 255, 0.2)',
            hovertemplate='<b>%{theta}</b><br>Importance: %{r}/10<br><extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(color='white', size=10),
                    gridcolor='rgba(255, 255, 255, 0.2)'
                ),
                angularaxis=dict(
                    tickfont=dict(color='white', size=12),
                    gridcolor='rgba(255, 255, 255, 0.2)'
                ),
                bgcolor='rgba(0, 0, 0, 0)'
            ),
            showlegend=False,
            title=dict(
                text="Skills Importance Radar",
                x=0.5,
                font=dict(color='#00c3ff', size=16)
            ),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig

    def get_course_description(course_title, role):
        """Generate a short description for the course"""
        descriptions = {
            'Frontend Developer': f"Master modern frontend development with {course_title.split()[0]} and build responsive web applications.",
            'Backend Developer': f"Learn server-side development and API design to become a skilled backend developer.",
            'Full Stack Developer': f"Comprehensive full-stack development course covering both frontend and backend technologies.",
            'Data Scientist': f"Dive deep into data science methodologies, machine learning, and statistical analysis.",
            'Machine Learning Engineer': f"Build and deploy machine learning models at scale with industry best practices.",
            'Cloud Architect': f"Design scalable cloud infrastructure and learn enterprise-grade cloud solutions.",
            'DevOps Engineer': f"Master CI/CD pipelines, containerization, and infrastructure automation.",
            'UI Designer': f"Create stunning user interfaces with modern design principles and tools.",
            'UX Designer': f"Learn user research, wireframing, and create exceptional user experiences."
        }
        
        return descriptions.get(role, f"Comprehensive course to advance your skills in {role} role.")

    def display_courses_by_difficulty(courses, role):
        """Display courses grouped by difficulty using index-based mapping"""
        # Group courses by difficulty
        difficulty_groups = {"Beginner": [], "Intermediate": [], "Advanced": []}
        
        for idx, (title, url) in enumerate(courses):
            difficulty = get_course_difficulty_by_index(idx)
            description = get_course_description(title, role)
            difficulty_groups[difficulty].append((title, url, description))
        
        # Display each difficulty group
        for difficulty in ["Beginner", "Intermediate", "Advanced"]:
            if difficulty_groups[difficulty]:
                st.markdown(f"### 🎯 {difficulty} Level")
                for title, url, description in difficulty_groups[difficulty]:
                    st.markdown(f"""
                        <div class="course-tile">
                            <div class="course-title">{title}</div>
                            <div class="course-description">{description}</div>
                            <span class="difficulty-badge difficulty-{difficulty.lower()}">{difficulty}</span>
                            <br>
                            <a href="{url}" target="_blank" class="course-link-btn">
                                🚀 Start Learning
                            </a>
                        </div>
                    """, unsafe_allow_html=True)

    # UPDATED SECTIONS

    # Section 1: UPDATED Courses by Role with Index-based Difficulty
    if page == "Courses by Role":
        st.subheader("🎯 Courses by Career Role")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Select Career Category",
                options=list(COURSES_BY_CATEGORY.keys()),
                key="category_selection"
            )
        
        with col2:
            if category:
                roles = list(COURSES_BY_CATEGORY[category].keys())
                role = st.selectbox(
                    "Select Role / Job Title",
                    options=roles,
                    key="role_selection"
                )
            else:
                role = None
        
        if category and role:
            # UPDATED: Add difficulty filter
            difficulty_filter = st.selectbox(
                "Filter by Difficulty Level",
                options=["All Levels", "Beginner", "Intermediate", "Advanced"],
                key="difficulty_filter"
            )
            
            st.subheader(f"📘 Courses for **{role}** in **{category}**:")
            courses = get_courses_for_role(category, role)
            
            if courses:
                # UPDATED: Display courses using index-based difficulty
                filtered_courses = []
                for idx, (title, url) in enumerate(courses):
                    difficulty = get_course_difficulty_by_index(idx)
                    
                    # Apply difficulty filter
                    if difficulty_filter == "All Levels" or difficulty == difficulty_filter:
                        filtered_courses.append((title, url, difficulty, idx))
                
                if filtered_courses:
                    for title, url, difficulty, idx in filtered_courses:
                        description = get_course_description(title, role)
                        
                        # UPDATED: Interactive course tile with index-based difficulty
                        st.markdown(f"""
                            <div class="course-tile">
                                <div class="course-title">{title}</div>
                                <div class="course-description">{description}</div>
                                <span class="difficulty-badge difficulty-{difficulty.lower()}">{difficulty}</span>
                                <br>
                                <a href="{url}" target="_blank" class="course-link-btn">
                                    🚀 Start Learning
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🚫 No courses found for this difficulty level.")
            else:
                st.info("🚫 No courses found for this role.")
        
        # Show skill radar chart for selected role
        if category and role:
            st.markdown("---")
            st.markdown('<div class="radar-container">', unsafe_allow_html=True)
            st.subheader("🎯 Skills Radar Chart")
            
            # Generate sample skills data based on role
            role_skills = {
                # ==== Software Development & Engineering ====
                "Frontend Developer": {
                    "JavaScript": 9, "React/Vue": 8, "CSS/HTML": 9,
                    "Responsive Design": 8, "Performance Optimization": 7, "Testing": 6
                },
                "Backend Developer": {
                    "API Design": 9, "Database Management": 8, "Security": 8,
                    "Scalability": 7, "Cloud Services": 7, "Testing": 6
                },
                "Full Stack Developer": {
                    "Frontend": 8, "Backend": 8, "Databases": 7,
                    "API Integration": 8, "DevOps Basics": 6, "Testing": 7
                },
                "Mobile App Developer": {
                    "Flutter/React Native": 8, "Swift/Kotlin": 8, "UI/UX": 8,
                    "APIs": 7, "Performance Optimization": 7, "App Deployment": 7
                },
                "Game Developer": {
                    "Unity/Unreal": 9, "C# / C++": 8, "Game Physics": 7,
                    "Graphics/Rendering": 8, "AI in Games": 6, "Multiplayer Systems": 7
                },
                # ==== Data Science & Analytics ====
                "Data Scientist": {
                    "Python/R": 9, "Machine Learning": 8, "Statistics": 9,
                    "Data Visualization": 7, "SQL": 8, "Domain Knowledge": 6
                },
                "Data Analyst": {
                    "SQL": 9, "Excel/Spreadsheets": 8, "Visualization": 8,
                    "Statistics": 8, "Python/R": 7, "Business Acumen": 7
                },
                "Machine Learning Engineer": {
                    "ML Algorithms": 9, "Deep Learning": 8, "MLOps": 7,
                    "Data Engineering": 8, "Python/Frameworks": 9, "Cloud Deployment": 7
                },
                # ==== Cloud Computing & DevOps ====
                "Cloud Architect": {
                    "AWS/Azure/GCP": 9, "System Design": 8, "Networking": 7,
                    "Security": 8, "Scalability": 9, "Cost Optimization": 7
                },
                "DevOps Engineer": {
                    "CI/CD": 9, "Containerization": 8, "Cloud Platforms": 8,
                    "Monitoring": 7, "Infrastructure as Code": 8, "Security": 7
                },
                "Site Reliability Engineer": {
                    "Reliability Engineering": 9, "Monitoring": 8, "Automation": 8,
                    "Incident Response": 8, "System Design": 7, "Security": 7
                },
                # ==== Cybersecurity ====
                "Security Analyst": {
                    "Threat Detection": 9, "Incident Response": 8, "Networking": 7,
                    "SIEM Tools": 8, "Risk Management": 7, "Compliance": 6
                },
                "Penetration Tester": {
                    "Ethical Hacking": 9, "Web Security": 8, "Exploitation": 8,
                    "Scripting": 7, "Reporting": 6, "Network Security": 7
                },
                # ==== UI/UX Design ====
                "UI Designer": {
                    "Design Tools": 9, "Visual Design": 8, "Typography": 7,
                    "Color Theory": 8, "Prototyping": 7, "User Research": 6
                },
                "UX Designer": {
                    "User Research": 9, "Wireframing": 8, "Prototyping": 8,
                    "Usability Testing": 7, "Accessibility": 8, "Design Thinking": 7
                },
                # ==== Project Management ====
                "Project Manager": {
                    "Planning": 9, "Communication": 8, "Risk Management": 8,
                    "Leadership": 7, "Agile/Scrum": 8, "Budgeting": 7
                },
                "Product Manager": {
                    "Market Research": 9, "Product Strategy": 8, "Analytics": 8,
                    "Communication": 8, "Agile Methods": 7, "User-Centered Design": 7
                }
            }
            
            skills_data = role_skills.get(role, {
                "Technical Skills": 8, "Problem Solving": 7, "Communication": 6,
                "Leadership": 5, "Domain Knowledge": 7, "Continuous Learning": 8
            })
            
            # Create and display radar chart
            radar_fig = create_skill_radar_chart(skills_data)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Add hover tooltip information
            st.markdown("""
                <div style="text-align: center; color: #00c3ff; margin-top: 10px;">
                    💡 Hover over the chart points to see skill importance ratings!
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Resume Videos (unchanged)
    elif page == "Resume Videos":
        st.subheader("📄 Resume Writing Videos")
        categories = list(RESUME_VIDEOS.keys())
        selected_cat = st.selectbox(
            "Select Resume Video Category",
            options=categories,
            key="resume_vid_cat"
        )
        if selected_cat:
            st.subheader(f"📂 {selected_cat}")
            videos = RESUME_VIDEOS[selected_cat]
            cols = st.columns(2)
            for idx, (title, url) in enumerate(videos):
                with cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.video(url)

    # Section 3: Interview Videos (unchanged)
    elif page == "Interview Videos":
        st.subheader("🗣️ Interview Preparation Videos")
        categories = list(INTERVIEW_VIDEOS.keys())
        selected_cat = st.selectbox(
            "Select Interview Video Category",
            options=categories,
            key="interview_vid_cat"
        )
        if selected_cat:
            st.subheader(f"📂 {selected_cat}")
            videos = INTERVIEW_VIDEOS[selected_cat]
            cols = st.columns(2)
            for idx, (title, url) in enumerate(videos):
                with cols[idx % 2]:
                    st.markdown(f"**{title}**")
                    st.video(url)

    # Section 4: UPDATED AI Interview Coach 🤖 with Resume-Based Interviewing
    elif page == "AI Interview Coach 🤖":
        st.subheader("🤖 AI Interview Coach")
        st.markdown("Upload your resume and practice role-specific interview questions with AI-powered feedback!")

        # Create database table if not exists
        create_interview_database()

        # Initialize resume state
        if 'resume_file' not in st.session_state:
            st.session_state.resume_file = None
        if 'resume_context' not in st.session_state:
            st.session_state.resume_context = None
        if 'interview_phase' not in st.session_state:
            st.session_state.interview_phase = "resume"
        if 'resume_questions_answered' not in st.session_state:
            st.session_state.resume_questions_answered = 0

        # RESUME UPLOAD SECTION (MANDATORY)
        st.markdown("---")
        st.markdown("<h3 style='color: #00c3ff;'>📄 Step 1: Upload Your Resume</h3>", unsafe_allow_html=True)

        if st.session_state.resume_file is None:
            uploaded_resume = st.file_uploader(
                "Upload your resume (PDF format)",
                type=['pdf'],
                key="resume_uploader"
            )

            if uploaded_resume:
                with st.spinner("Processing your resume..."):
                    # Extract text from PDF
                    resume_text = extract_resume_text_from_pdf(uploaded_resume)

                    if resume_text and len(resume_text.strip()) > 50:
                        # Analyze resume
                        with st.spinner("Analyzing your resume with AI..."):
                            resume_context = analyze_resume_with_llm(resume_text)

                        # Store in session
                        st.session_state.resume_file = uploaded_resume.name
                        st.session_state.resume_context = resume_context
                        st.session_state.interview_phase = "resume"
                        st.session_state.resume_questions_answered = 0

                        st.success("✅ Resume uploaded and analyzed successfully!")
                        
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Could not extract text from resume. Please ensure it's a valid PDF.")
        else:
            st.success(f"✅ Resume loaded: {st.session_state.resume_file}")
            

            if st.button("🔄 Upload Different Resume"):
                st.session_state.resume_file = None
                st.session_state.resume_context = None
                st.session_state.dynamic_interview_started = False
                st.session_state.dynamic_interview_completed = False
                st.rerun()

        # Only show domain/role selection if resume is uploaded
        if st.session_state.resume_file is not None:
            st.markdown("---")
            st.markdown("<h3 style='color: #00c3ff;'>👔 Step 2: Select Target Role</h3>", unsafe_allow_html=True)

            # Domain and Role selection
            st.markdown('<div class="role-selector">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                selected_domain = st.selectbox(
                    "Select Career Domain",
                    options=list(COURSES_BY_CATEGORY.keys()),
                    key="interview_domain_selection"
                )

            with col2:
                if selected_domain:
                    roles = list(COURSES_BY_CATEGORY[selected_domain].keys())
                    selected_role = st.selectbox(
                        "Select Target Role",
                        options=roles,
                        key="interview_role_selection"
                    )
                else:
                    selected_role = None

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_domain = None
            selected_role = None
        
        if selected_domain and selected_role:
            # Initialize interview state
            if 'dynamic_interview_questions' not in st.session_state:
                st.session_state.dynamic_interview_questions = []
            if 'current_dynamic_interview_question' not in st.session_state:
                st.session_state.current_dynamic_interview_question = 0
            if 'dynamic_interview_answers' not in st.session_state:
                st.session_state.dynamic_interview_answers = []
            if 'dynamic_interview_scores' not in st.session_state:
                st.session_state.dynamic_interview_scores = []
            if 'dynamic_interview_feedbacks' not in st.session_state:
                st.session_state.dynamic_interview_feedbacks = []
            if 'dynamic_interview_completed' not in st.session_state:
                st.session_state.dynamic_interview_completed = False
            if 'dynamic_interview_started' not in st.session_state:
                st.session_state.dynamic_interview_started = False
            if 'dynamic_answer_submitted' not in st.session_state:
                st.session_state.dynamic_answer_submitted = False
            if 'current_interview_question_text' not in st.session_state:
                st.session_state.current_interview_question_text = ""
            if 'interview_domain' not in st.session_state or st.session_state.interview_domain != selected_domain:
                st.session_state.interview_domain = selected_domain
                st.session_state.interview_role = selected_role
                st.session_state.dynamic_interview_started = False
                st.session_state.dynamic_interview_completed = False
            if 'question_timer_start' not in st.session_state:
                st.session_state.question_timer_start = None
            if 'timer_seconds' not in st.session_state:
                st.session_state.timer_seconds = 120
            if 'interview_difficulty' not in st.session_state:
                st.session_state.interview_difficulty = "Medium"
            if 'original_num_questions' not in st.session_state:
                st.session_state.original_num_questions = 6
            if 'resume_based_questions' not in st.session_state:
                st.session_state.resume_based_questions = []
            if 'generic_questions' not in st.session_state:
                st.session_state.generic_questions = []

            # Start interview setup
            if not st.session_state.dynamic_interview_started:
                st.markdown(f"### Practice interview for: {selected_role}")

                col1, col2 = st.columns(2)

                with col1:
                    interview_type = st.selectbox(
                        "Interview Type",
                        options=["technical", "behavioral", "mixed"],
                        format_func=lambda x: x.title() + (" (Technical + Behavioral)" if x == "mixed" else ""),
                        key="dynamic_interview_type_select"
                    )

                with col2:
                    interview_difficulty = st.selectbox(
                        "Interview Difficulty",
                        options=["Easy", "Medium", "Hard"],
                        key="interview_difficulty_select",
                        index=1
                    )

                col3, col4 = st.columns(2)
                with col3:
                    num_questions = st.slider("Number of questions:", 5, 10, 6)

                with col4:
                    timer_seconds = st.slider("Time per question (seconds):", 60, 300, 120, step=30)

                if st.button("🚀 Start Mock Interview"):
                    with st.spinner("Generating personalized questions using AI..."):
                        # Generate resume-based questions
                        resume_based_qs = []
                        if st.session_state.resume_context:
                            with st.spinner("Creating resume-based questions..."):
                                resume_based_qs = generate_resume_based_questions(
                                    st.session_state.resume_context,
                                    selected_role,
                                    selected_domain,
                                    interview_difficulty,
                                    num_questions=2
                                )

                        # Generate generic questions
                        generic_qs = []
                        remaining_questions = num_questions - len(resume_based_qs)
                        if remaining_questions > 0:
                            with st.spinner("Creating generic interview questions..."):
                                generic_qs = generate_interview_questions_with_llm(
                                    selected_domain,
                                    selected_role,
                                    interview_type,
                                    remaining_questions,
                                    interview_difficulty
                                )

                        # Combine all questions: resume-based first, then generic
                        all_questions = resume_based_qs + generic_qs
                        all_questions = all_questions[:num_questions]

                        if all_questions:
                            # Reset ALL interview state variables properly
                            st.session_state.dynamic_interview_questions = all_questions
                            st.session_state.resume_based_questions = resume_based_qs
                            st.session_state.generic_questions = generic_qs
                            st.session_state.original_num_questions = num_questions
                            st.session_state.current_dynamic_interview_question = 0
                            st.session_state.dynamic_interview_answers = []
                            st.session_state.dynamic_interview_scores = []
                            st.session_state.dynamic_interview_feedbacks = []
                            st.session_state.dynamic_interview_completed = False
                            st.session_state.dynamic_interview_started = True
                            st.session_state.dynamic_answer_submitted = False
                            st.session_state.current_interview_question_text = all_questions[0]
                            st.session_state.question_timer_start = time.time()
                            st.session_state.timer_seconds = timer_seconds
                            st.session_state.interview_difficulty = interview_difficulty
                            st.session_state.interview_phase = "resume" if resume_based_qs else "generic"

                            # Show resume scanning animation if resume questions exist
                            if resume_based_qs:
                                st.info("🎯 Starting with resume-based questions...")
                                show_resume_scanning_animation()

                            st.success("Questions generated! Starting your mock interview...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to generate questions. Please try again.")
            
            # Interview in progress
            elif st.session_state.dynamic_interview_started and not st.session_state.dynamic_interview_completed:
                # CRITICAL FIX: Properly count answered questions
                questions_answered = len(st.session_state.dynamic_interview_answers)
                total_questions = len(st.session_state.dynamic_interview_questions)
                current_index = st.session_state.current_dynamic_interview_question + 1

                # Determine current phase
                num_resume_qs = len(st.session_state.resume_based_questions)
                current_phase = "Resume-Based" if current_index <= num_resume_qs else "Generic Interview"

                # Display progress with correct counts in glassmorphism box
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0, 195, 255, 0.08) 0%, rgba(0, 195, 255, 0.04) 100%);
                            backdrop-filter: blur(10px);
                            -webkit-backdrop-filter: blur(10px);
                            border: 1px solid rgba(0, 195, 255, 0.2);
                            border-radius: 12px;
                            padding: 16px 24px;
                            margin: 20px 0;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);">
                    <p style="color: #ffffff; font-size: 16px; margin: 0; font-weight: 500;">
                        📊 Progress: Answered {questions_answered}/{st.session_state.original_num_questions} questions | Phase: {current_phase}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if questions_answered < st.session_state.original_num_questions:
                    question = st.session_state.current_interview_question_text or st.session_state.dynamic_interview_questions[st.session_state.current_dynamic_interview_question]

                    # TIMER RESET: Reset timer every time a new question loads
                    if st.session_state.question_timer_start is None:
                        st.session_state.question_timer_start = time.time()

                    # Calculate remaining time
                    elapsed_time = time.time() - st.session_state.question_timer_start
                    remaining_time = max(0, st.session_state.timer_seconds - elapsed_time)

                    # Display timer
                    timer_minutes = int(remaining_time // 60)
                    timer_seconds_display = int(remaining_time % 60)
                    timer_urgent_class = "timer-urgent" if remaining_time <= 30 else ""

                    st.markdown(f"""
                    <div class="timer-container">
                        <div class="timer-display {timer_urgent_class}">
                            ⏰ Time Remaining: {timer_minutes:02d}:{timer_seconds_display:02d}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Timer progress bar
                    progress_value = (st.session_state.timer_seconds - remaining_time) / st.session_state.timer_seconds
                    st.progress(progress_value)

                    # Question display with phase indicator
                    phase_badge = "📄 Resume-Based Question" if current_index <= num_resume_qs else "💼 Generic Interview Question"
                    st.markdown(f"""
                    <div class="quiz-card">
                        <h3 style="color: #00c3ff;">Question {questions_answered + 1} of {st.session_state.original_num_questions}</h3>
                        <div style="background: rgba(0, 195, 255, 0.15); padding: 8px 12px; border-radius: 6px; margin: 10px 0; display: inline-block;">
                            <span style="color: #00c3ff; font-weight: 600;">{phase_badge}</span>
                        </div>
                        <h4 style="color: #ffffff; margin: 15px 0;">Role: {selected_role} | Difficulty: {st.session_state.interview_difficulty}</h4>
                        <p style="font-size: 18px; color: #ffffff; margin: 15px 0;">{question}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add refresh button for regenerating all interview questions
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("🔄 Refresh Interview"):
                            # Clear all interview state
                            st.session_state.dynamic_interview_questions = []
                            st.session_state.current_dynamic_interview_question = 0
                            st.session_state.dynamic_interview_answers = []
                            st.session_state.dynamic_interview_scores = []
                            st.session_state.dynamic_interview_feedbacks = []
                            st.session_state.dynamic_interview_completed = False
                            st.session_state.dynamic_interview_started = False
                            st.session_state.dynamic_answer_submitted = False
                            st.session_state.current_interview_question_text = ""
                            st.session_state.question_timer_start = None
                            # Force regeneration
                            st.rerun()

                    # Answer input with character limit
                    answer_key = f"dynamic_interview_answer_{st.session_state.current_dynamic_interview_question}"
                    answer = st.text_area(
                        "Your answer:",
                        placeholder="Type your detailed answer here... (Use STAR method: Situation, Task, Action, Result)",
                        height=150,
                        max_chars=2000,
                        key=answer_key,
                        help="Maximum 2000 characters"
                    )

                    # Auto-submit logic when timer expires
                    if remaining_time <= 0 and not st.session_state.dynamic_answer_submitted:
                        if not answer.strip():
                            answer = "⚠️ No Answer"

                        # Evaluate answer using enhanced evaluation with role/domain context
                        with st.spinner("Evaluating your answer..."):
                            eval_result = evaluate_interview_answer_for_scores(
                                answer,
                                question,
                                st.session_state.interview_difficulty,
                                role=selected_role,
                                domain=selected_domain
                            )

                        # FIXED: Store answer, scores, and feedback - ensuring all are tracked properly
                        st.session_state.dynamic_interview_answers.append(answer)
                        st.session_state.dynamic_interview_scores.append(eval_result)
                        st.session_state.dynamic_interview_feedbacks.append(eval_result["feedback"])
                        st.session_state.dynamic_answer_submitted = True

                        # FIXED: Handle follow-up for Hard difficulty without breaking indexing
                        # Follow-ups are added but don't count toward original_num_questions
                        if st.session_state.interview_difficulty == "Hard" and eval_result.get("followup") and eval_result["followup"].strip():
                            # Only add follow-up if we haven't reached the end
                            if questions_answered < st.session_state.original_num_questions - 1:
                                st.session_state.dynamic_interview_questions.insert(
                                    st.session_state.current_dynamic_interview_question + 1,
                                    eval_result["followup"]
                                )

                        st.warning("⏰ Time's up! Answer auto-submitted.")
                        st.rerun()

                    # Submit answer button
                    if not st.session_state.dynamic_answer_submitted and remaining_time > 0:
                        if st.button("Submit Answer & Get Feedback"):
                            if answer.strip():
                                with st.spinner("Evaluating your answer..."):
                                    # Evaluate answer using enhanced evaluation with role/domain context
                                    eval_result = evaluate_interview_answer_for_scores(
                                        answer,
                                        question,
                                        st.session_state.interview_difficulty,
                                        role=selected_role,
                                        domain=selected_domain
                                    )

                                    # FIXED: Store answer, scores, and feedback ensuring proper tracking
                                    st.session_state.dynamic_interview_answers.append(answer)
                                    st.session_state.dynamic_interview_scores.append(eval_result)
                                    st.session_state.dynamic_interview_feedbacks.append(eval_result["feedback"])
                                    st.session_state.dynamic_answer_submitted = True

                                    # FIXED: Handle follow-up for Hard difficulty without breaking indexing
                                    if st.session_state.interview_difficulty == "Hard" and eval_result.get("followup") and eval_result["followup"].strip():
                                        # Only add follow-up if we haven't reached the end
                                        if questions_answered < st.session_state.original_num_questions - 1:
                                            st.session_state.dynamic_interview_questions.insert(
                                                st.session_state.current_dynamic_interview_question + 1,
                                                eval_result["followup"]
                                            )

                                    st.rerun()
                            else:
                                st.warning("Please provide an answer before proceeding.")

                    # Show feedback after answer submitted
                    if st.session_state.dynamic_answer_submitted:
                        current_score_dict = st.session_state.dynamic_interview_scores[-1]
                        avg_q_score = (current_score_dict["knowledge"] + current_score_dict["communication"] + current_score_dict["relevance"]) / 3

                        # Format feedback for display
                        feedback_text = current_score_dict["feedback"] if isinstance(current_score_dict["feedback"], str) else chr(10).join(current_score_dict["feedback"])
                        formatted_feedback = format_feedback_text(feedback_text)

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(0, 195, 255, 0.1) 0%, rgba(0, 195, 255, 0.05) 100%);
                                    border: 1px solid rgba(0, 195, 255, 0.3); border-radius: 10px; padding: 15px; margin: 15px 0;">
                            <h4 style="color: #00c3ff;">Immediate Feedback:</h4>
                            <p style="color: #ffffff;">📊 Knowledge: {current_score_dict["knowledge"]}/10 | Communication: {current_score_dict["communication"]}/10 | Relevance: {current_score_dict["relevance"]}/10</p>
                            <p style="color: #ffffff;">⭐ Question Score: {avg_q_score:.1f}/10</p>
                            <div style="color: #ffffff; margin-top: 10px;">
                                {formatted_feedback}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show follow-up question for Hard difficulty
                        if st.session_state.interview_difficulty == "Hard" and current_score_dict.get("followup"):
                            st.info(f"🔎 Follow-Up Question: {current_score_dict['followup']}")

                        # Continue/Complete button
                        # CRITICAL FIX: Check if we've answered all original questions
                        if questions_answered >= st.session_state.original_num_questions:
                            # All questions answered, mark as complete
                            if st.button("Complete Interview 🏁"):
                                st.session_state.dynamic_interview_completed = True
                                st.rerun()
                        else:
                            # More questions to go
                            if st.button("Continue to Next Question ➡️"):
                                st.session_state.current_dynamic_interview_question += 1
                                st.session_state.dynamic_answer_submitted = False
                                if st.session_state.current_dynamic_interview_question < len(st.session_state.dynamic_interview_questions):
                                    st.session_state.current_interview_question_text = st.session_state.dynamic_interview_questions[st.session_state.current_dynamic_interview_question]
                                else:
                                    # Safety check - if we're out of questions but haven't answered all, generate one
                                    st.session_state.current_interview_question_text = f"Additional question for {selected_role}"
                                # TIMER RESET: Reset timer for next question
                                st.session_state.question_timer_start = time.time()
                                st.rerun()

                    # Progress bar for interview completion
                    interview_progress = questions_answered / st.session_state.original_num_questions
                    st.markdown("### Interview Progress")
                    st.progress(interview_progress)

                    # CRITICAL FIX: Review Previous Answers - show all properly
                    if len(st.session_state.dynamic_interview_answers) > 0:
                        with st.expander("📖 Review Previous Answers"):
                            # Show all submitted answers
                            num_to_show = len(st.session_state.dynamic_interview_answers)
                            for i in range(num_to_show):
                                if i < len(st.session_state.dynamic_interview_questions) and i < len(st.session_state.dynamic_interview_scores):
                                    prev_question = st.session_state.dynamic_interview_questions[i]
                                    prev_answer = st.session_state.dynamic_interview_answers[i]
                                    prev_scores = st.session_state.dynamic_interview_scores[i]
                                    prev_avg = (prev_scores["knowledge"] + prev_scores["communication"] + prev_scores["relevance"]) / 3

                                    # Show full answer (up to 500 chars in review, full in final)
                                    answer_preview = prev_answer[:500]
                                    if len(prev_answer) > 500:
                                        answer_preview += "..."

                                    st.markdown(f"**Question {i+1}:** {prev_question}")
                                    st.markdown(f"**Your Answer:** {answer_preview}")
                                    st.markdown(f"**Score:** {prev_avg:.1f}/10")
                                    if i < num_to_show - 1:  # Don't add separator after last item
                                        st.markdown("---")

                    # Auto-refresh for timer
                    if remaining_time > 0 and not st.session_state.dynamic_answer_submitted:
                        time.sleep(1)
                        st.rerun()
                else:
                    # CRITICAL FIX: All questions answered, move to completion automatically
                    st.session_state.dynamic_interview_completed = True
                    st.success(f"✅ Completed all {st.session_state.original_num_questions} questions!")
                    time.sleep(1)
                    st.rerun()
            
            # UNIFIED: Interview completed + Course Recommendations + DB + PDF
            elif st.session_state.dynamic_interview_completed:
                # Calculate average scores for each dimension
                knowledge_scores = [s["knowledge"] for s in st.session_state.dynamic_interview_scores]
                communication_scores = [s["communication"] for s in st.session_state.dynamic_interview_scores]
                relevance_scores = [s["relevance"] for s in st.session_state.dynamic_interview_scores]

                avg_knowledge = sum(knowledge_scores) / len(knowledge_scores)
                avg_communication = sum(communication_scores) / len(communication_scores)
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                overall_avg = (avg_knowledge + avg_communication + avg_relevance) / 3

                # Determine badge based on overall average
                if overall_avg >= 8.5:
                    badge = "Interview Ready"
                    badge_emoji = "🏆"
                elif overall_avg >= 7.0:
                    badge = "Excellent"
                    badge_emoji = "🌟"
                elif overall_avg >= 5.0:
                    badge = "Good"
                    badge_emoji = "👍"
                else:
                    badge = "Needs Practice"
                    badge_emoji = "💪"

                st.markdown(f"""
                <div class="badge-container">
                    <h2 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 600;">🎉 Mock Interview Complete!</h2>
                    <div style="margin: 30px 0;">
                        <div class="score-display">{overall_avg:.1f}/10</div>
                        <h3 style="color: #ffffff; margin: 15px 0; font-size: 24px; font-weight: 500;">{badge_emoji} {badge}</h3>
                    </div>
                    <p style="color: rgba(255, 255, 255, 0.85); font-size: 16px; margin: 8px 0;">Role: {selected_role} in {selected_domain}</p>
                    <p style="color: rgba(255, 255, 255, 0.85); font-size: 16px; margin: 8px 0;">Difficulty: {st.session_state.interview_difficulty}</p>
                </div>
                """, unsafe_allow_html=True)

                # Create radar chart for skills
                st.markdown('<div class="radar-container">', unsafe_allow_html=True)
                st.subheader("📊 Performance Radar Chart")

                radar_data = {
                    "Communication": avg_communication,
                    "Knowledge": avg_knowledge,
                    "Confidence": avg_relevance
                }

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(radar_data.values()),
                    theta=list(radar_data.keys()),
                    fill='toself',
                    name='Performance',
                    line=dict(color='#00c3ff', width=2),
                    fillcolor='rgba(0, 195, 255, 0.2)'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10],
                            tickfont=dict(color='white', size=10),
                            gridcolor='rgba(255, 255, 255, 0.2)'
                        ),
                        angularaxis=dict(
                            tickfont=dict(color='white', size=12),
                            gridcolor='rgba(255, 255, 255, 0.2)'
                        ),
                        bgcolor='rgba(0, 0, 0, 0)'
                    ),
                    showlegend=False,
                    title=dict(
                        text="Interview Performance Metrics",
                        x=0.5,
                        font=dict(color='#00c3ff', size=16)
                    ),
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(color='white'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Strengths and Weaknesses
                st.subheader("💡 Performance Analysis")
                col1, col2 = st.columns(2)

                metrics = [("Communication", avg_communication), ("Knowledge", avg_knowledge), ("Confidence", avg_relevance)]
                metrics_sorted = sorted(metrics, key=lambda x: x[1], reverse=True)

                with col1:
                    st.markdown("**🌟 Strengths:**")
                    for name, score in metrics_sorted[:2]:
                        st.markdown(f"- {name}: {score:.1f}/10")

                with col2:
                    st.markdown("**📈 Areas to Improve:**")
                    for name, score in metrics_sorted[-2:]:
                        st.markdown(f"- {name}: {score:.1f}/10")

                # FIXED: Show detailed Q&A results with full answers and proper matching
                st.markdown("---")
                st.subheader("📋 Detailed Q&A Review:")

                # Ensure we only show as many Q&A pairs as we have complete data for
                num_complete_qa = min(
                    len(st.session_state.dynamic_interview_scores),
                    len(st.session_state.dynamic_interview_answers),
                    len(st.session_state.dynamic_interview_feedbacks),
                    len(st.session_state.dynamic_interview_questions)
                )

                for i in range(num_complete_qa):
                    score_dict = st.session_state.dynamic_interview_scores[i]
                    answer = st.session_state.dynamic_interview_answers[i]
                    feedback = st.session_state.dynamic_interview_feedbacks[i]
                    question = st.session_state.dynamic_interview_questions[i]

                    q_avg = (score_dict["knowledge"] + score_dict["communication"] + score_dict["relevance"]) / 3

                    with st.expander(f"Question {i+1}: Score {q_avg:.1f}/10"):
                        st.write(f"**Question:** {question}")
                        st.write(f"**Your Answer:** {answer}")  # Show full answer
                        st.write(f"**Scores:** Knowledge: {score_dict['knowledge']}/10 | Communication: {score_dict['communication']}/10 | Relevance: {score_dict['relevance']}/10")

                        # Format and display feedback as bullet points
                        feedback_text = "\n".join(feedback) if isinstance(feedback, list) else feedback
                        formatted_feedback = format_feedback_text(feedback_text)
                        st.markdown(formatted_feedback, unsafe_allow_html=True)

                # Save to database
                username = st.session_state.get("username", "Guest")
                feedback_summary = f"Strengths: {metrics_sorted[0][0]}, {metrics_sorted[1][0]}. Weaknesses: {metrics_sorted[-1][0]}, {metrics_sorted[-2][0]}."

                if save_interview_result(username, selected_role, selected_domain, overall_avg, st.session_state.original_num_questions, feedback_summary):
                    log_user_action(username, "completed_interview")

                # Generate PDF report
                st.markdown("---")
                st.subheader("📄 Download Interview Report")

                completed_on = get_ist_time()

                # CRITICAL FIX: Ensure all arrays have same length for PDF generation
                num_complete = min(
                    len(st.session_state.dynamic_interview_questions),
                    len(st.session_state.dynamic_interview_answers),
                    len(st.session_state.dynamic_interview_scores),
                    len(st.session_state.dynamic_interview_feedbacks)
                )

                pdf_bytes = generate_interview_pdf_report(
                    username,
                    selected_role,
                    selected_domain,
                    completed_on,
                    st.session_state.dynamic_interview_questions[:num_complete],
                    st.session_state.dynamic_interview_answers[:num_complete],
                    st.session_state.dynamic_interview_scores[:num_complete],
                    st.session_state.dynamic_interview_feedbacks[:num_complete],
                    overall_avg,
                    badge,
                    difficulty=st.session_state.interview_difficulty
                )

                if pdf_bytes:
                    st.download_button(
                        label="📄 Download Interview Report",
                        data=pdf_bytes,
                        file_name=f"interview_report_{username}_{selected_role.replace(' ', '_')}_{completed_on.split()[0]}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("PDF generation failed. You can still review your results above.")

                # UNIFIED: Display recommended courses by difficulty
                st.markdown("---")
                st.subheader("📚 Recommended Courses for Your Career Growth")
                st.markdown(f"Based on your interview practice for **{selected_role}** in **{selected_domain}**, here are our course recommendations organized by difficulty level:")

                courses = get_courses_for_role(selected_domain, selected_role)
                if courses:
                    display_courses_by_difficulty(courses, selected_role)
                else:
                    st.info("No specific courses found for this role. Explore our course categories to find relevant learning resources!")

                # FIXED: Restart button - properly resets ALL interview state
                if st.button("🔄 Practice Again"):
                    # Reset all interview-related session state variables
                    st.session_state.dynamic_interview_started = False
                    st.session_state.dynamic_interview_completed = False
                    st.session_state.dynamic_interview_questions = []
                    st.session_state.current_dynamic_interview_question = 0
                    st.session_state.dynamic_interview_answers = []
                    st.session_state.dynamic_interview_scores = []
                    st.session_state.dynamic_interview_feedbacks = []
                    st.session_state.dynamic_answer_submitted = False
                    st.session_state.current_interview_question_text = ""
                    st.session_state.question_timer_start = None
                    st.session_state.timer_seconds = 120
                    st.session_state.interview_difficulty = "Medium"
                    st.session_state.original_num_questions = 6
                    st.session_state.resume_based_questions = []
                    st.session_state.generic_questions = []
                    st.session_state.interview_phase = "resume"
                    st.rerun()
        else:
            st.info("Please select both a career domain and target role to start the interview practice.")