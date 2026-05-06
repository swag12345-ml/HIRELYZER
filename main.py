"""
TAB_JOB_SCAM_DETECTOR.py  —  HIRELYZER  |  Job Scam Detector
═══════════════════════════════════════════════════════════════
Advanced AI-powered job scam detection with:
  • Real-time scam probability scoring (0–100)
  • Multi-signal red flag extraction (salary, urgency, grammar, links, company)
  • Side-by-side fake vs. real job description comparison
  • Company legitimacy intelligence check
  • Scam pattern library & education
  • Rate-limited per user (feature: "job_scam_detector")
  • Full session history with per-analysis details
  • Export analysis as PDF / copy to clipboard
  • Integrates with user_login.py  &  llm_manager.py
"""

import json
import re
import time
from datetime import datetime

import pytz
import streamlit as st

# ── Internal imports (same pattern as other tabs) ─────────────────────────────
try:
    from llm_manager import call_llm
except ImportError:
    def call_llm(prompt, session, **kw):
        return '{"error": "LLM unavailable"}'

try:
    from user_login import check_and_gate_feature, record_feature_usage, get_user_email_by_username
except ImportError:
    def check_and_gate_feature(u, f): return True, "ok"
    def record_feature_usage(u, f): pass
    def get_user_email_by_username(u): return ""

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_KEY = "job_scam_detector"

SCAM_PATTERNS = {
    "Unrealistic Salary": {
        "icon": "💰",
        "color": "#ef4444",
        "desc": "Promises far above market rate for the role/experience required.",
        "examples": ["$5000/week from home", "Earn $200/hr no experience needed"],
    },
    "Vague Job Description": {
        "icon": "🌫️",
        "color": "#f97316",
        "desc": "No specific duties, responsibilities, or required skills listed.",
        "examples": ["Be your own boss!", "Flexible work — details on joining"],
    },
    "Upfront Payment Required": {
        "icon": "💳",
        "color": "#ef4444",
        "desc": "Asks you to pay for training, kits, background checks, or registration.",
        "examples": ["Pay $50 starter kit", "Purchase equipment to begin"],
    },
    "Personal Info Too Early": {
        "icon": "🪪",
        "color": "#ef4444",
        "desc": "Requests SSN, bank details, or passport before any formal process.",
        "examples": ["Send us your bank account", "Attach ID proof with application"],
    },
    "Urgency Pressure": {
        "icon": "⏱️",
        "color": "#f97316",
        "desc": "Creates false urgency to prevent you from thinking critically.",
        "examples": ["Limited seats! Apply in 2 hours", "Offer expires TODAY"],
    },
    "No Company Details": {
        "icon": "🏚️",
        "color": "#f97316",
        "desc": "Missing company name, address, registration number, or official website.",
        "examples": ["A top MNC is hiring", "Global company seeks candidates"],
    },
    "Suspicious Contact Method": {
        "icon": "📲",
        "color": "#eab308",
        "desc": "Uses WhatsApp, Telegram, personal Gmail, or random phone numbers.",
        "examples": ["WhatsApp: +91-XXXXXX", "Email: hr_jobs_2024@gmail.com"],
    },
    "Grammar & Spelling Errors": {
        "icon": "✏️",
        "color": "#eab308",
        "desc": "Excessive typos, broken English, or inconsistent formatting.",
        "examples": ["We are seeking candidate for work.", "Salary upto 5 lakhs + benifits"],
    },
    "Work From Home With No Skills": {
        "icon": "🏠",
        "color": "#eab308",
        "desc": "Promises high WFH salary requiring zero qualifications or training.",
        "examples": ["Work 2hrs/day, earn ₹50k/month", "No skills needed, full training"],
    },
    "Too Good To Be True": {
        "icon": "🌟",
        "color": "#a855f7",
        "desc": "Overall offer significantly exceeds market norms for the role.",
        "examples": ["Freshers earn ₹1L/month", "Part time job pays more than your full time"],
    },
}

EXAMPLE_REAL_JD = """Software Engineer — Backend (Python)
Company: Infosys BPM Ltd. | Pune, Maharashtra
Salary: ₹8–12 LPA | Experience: 2–4 years

We are hiring Backend Engineers for our digital transformation team.

Responsibilities:
• Design and develop RESTful APIs using Python (FastAPI / Django REST)
• Work with PostgreSQL and Redis for data layer design
• Participate in Agile sprints, code reviews, and architecture discussions
• Collaborate with frontend and DevOps teams

Requirements:
• B.Tech/B.E. in CS, IT, or related field
• 2+ years of Python backend development
• Familiarity with Docker and CI/CD pipelines
• Strong SQL fundamentals

Benefits:
• Health insurance (self + family)
• 24 days paid leave
• Learning & certification budget

Apply at careers.infosys.com/job/12345 or email recruitment@infosys.com
CIN: U72200MH2009PLC193116"""

EXAMPLE_FAKE_JD = """URGENT HIRING!!! 🔥🔥🔥
WORK FROM HOME | EARN ₹80,000 PER MONTH

A leading MNC is hiring data entry executives. No experience needed!

✅ Freshers welcome
✅ Only 2 hours of work daily
✅ Weekly payments guaranteed
✅ FREE laptop provided on joining

Job Duties: Simple copy-paste tasks

To apply: Send your CV + Aadhar + Bank account details to hr.jobs2024india@gmail.com
OR WhatsApp: +91-98XXXXXXX

Hurry! Only 5 seats left. Offer valid for 24 HOURS only.

Registration fee: ₹500 (refundable after first payment)"""

# ═══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
  --scam-red:    #ef4444;
  --scam-orange: #f97316;
  --scam-yellow: #eab308;
  --safe-green:  #22c55e;
  --neutral:     #94a3b8;
  --bg-glass:    rgba(15,23,42,0.6);
  --border-glow: rgba(239,68,68,0.3);
  --text-primary: #f1f5f9;
  --text-muted:   #94a3b8;
  --surface:      rgba(30,41,59,0.8);
}

/* ── Score meter ── */
.scam-meter-wrap {
  background: var(--surface);
  border: 1px solid rgba(148,163,184,0.15);
  border-radius: 16px;
  padding: 28px 32px;
  margin: 16px 0;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
}
.scam-meter-wrap::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at top right, rgba(239,68,68,0.05), transparent 60%);
  pointer-events: none;
}
.scam-score-number {
  font-family: 'Syne', sans-serif;
  font-size: 72px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -2px;
}
.scam-score-label {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-top: 4px;
}
.scam-bar-track {
  width: 100%;
  height: 10px;
  background: rgba(148,163,184,0.15);
  border-radius: 99px;
  margin: 18px 0 8px;
  overflow: hidden;
}
.scam-bar-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 1.2s cubic-bezier(0.34,1.56,0.64,1);
}

/* ── Verdict badge ── */
.verdict-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 20px;
  border-radius: 99px;
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}

/* ── Red flag cards ── */
.flag-card {
  background: rgba(239,68,68,0.06);
  border: 1px solid rgba(239,68,68,0.2);
  border-left: 3px solid var(--scam-red);
  border-radius: 10px;
  padding: 12px 16px;
  margin: 8px 0;
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  color: #fca5a5;
}
.flag-card.warn {
  background: rgba(249,115,22,0.06);
  border-color: rgba(249,115,22,0.2);
  border-left-color: var(--scam-orange);
  color: #fdba74;
}
.flag-card.caution {
  background: rgba(234,179,8,0.06);
  border-color: rgba(234,179,8,0.2);
  border-left-color: var(--scam-yellow);
  color: #fde047;
}
.flag-card.safe {
  background: rgba(34,197,94,0.06);
  border-color: rgba(34,197,94,0.2);
  border-left-color: var(--safe-green);
  color: #86efac;
}
.flag-title {
  font-weight: 500;
  margin-bottom: 3px;
}
.flag-desc {
  font-size: 11.5px;
  opacity: 0.75;
  font-family: 'DM Mono', monospace;
}

/* ── Compare cards ── */
.compare-card {
  border-radius: 14px;
  padding: 20px;
  height: 100%;
  font-family: 'DM Mono', monospace;
  font-size: 12.5px;
  line-height: 1.7;
}
.compare-card.fake {
  background: rgba(239,68,68,0.07);
  border: 1px solid rgba(239,68,68,0.25);
}
.compare-card.real {
  background: rgba(34,197,94,0.07);
  border: 1px solid rgba(34,197,94,0.25);
}
.compare-card h4 {
  font-family: 'Syne', sans-serif;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin: 0 0 14px;
}
.compare-card.fake h4 { color: #fca5a5; }
.compare-card.real h4 { color: #86efac; }

/* ── Pattern pill ── */
.pattern-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 99px;
  font-size: 11.5px;
  font-family: 'DM Mono', monospace;
  margin: 3px;
  cursor: pointer;
  transition: transform 0.15s;
}
.pattern-pill:hover { transform: translateY(-1px); }

/* ── Section header ── */
.section-hdr {
  font-family: 'Syne', sans-serif;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 24px 0 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(148,163,184,0.1);
}

/* ── History item ── */
.hist-item {
  background: var(--surface);
  border: 1px solid rgba(148,163,184,0.1);
  border-radius: 10px;
  padding: 12px 16px;
  margin: 6px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
}

/* ── Info callout ── */
.info-callout {
  background: rgba(99,102,241,0.08);
  border: 1px solid rgba(99,102,241,0.2);
  border-radius: 10px;
  padding: 14px 18px;
  font-family: 'DM Mono', monospace;
  font-size: 12.5px;
  color: #a5b4fc;
  margin: 10px 0;
}

/* ── Tab heading ── */
.tab-hero {
  text-align: center;
  padding: 32px 0 8px;
}
.tab-hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: 38px;
  font-weight: 800;
  letter-spacing: -1px;
  background: linear-gradient(135deg, #ef4444 0%, #f97316 50%, #eab308 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
}
.tab-hero p {
  color: var(--text-muted);
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_analysis_prompt(jd_text: str) -> str:
    patterns = ", ".join(SCAM_PATTERNS.keys())
    return f"""You are an expert job scam detection system. Analyse the following job description and return ONLY valid JSON (no markdown, no extra text).

Job Description:
\"\"\"{jd_text}\"\"\"

Return this exact JSON structure:
{{
  "scam_score": <integer 0-100, where 0=definitely legitimate, 100=definite scam>,
  "verdict": "<one of: SAFE | SUSPICIOUS | LIKELY SCAM | DEFINITE SCAM>",
  "verdict_reason": "<one sentence summary>",
  "red_flags": [
    {{
      "flag": "<flag name from: {patterns}>",
      "severity": "<HIGH | MEDIUM | LOW>",
      "evidence": "<exact text from the JD that triggered this flag>",
      "explanation": "<why this is a red flag>"
    }}
  ],
  "positive_signals": [
    "<signal 1>",
    "<signal 2>"
  ],
  "company_legitimacy": {{
    "score": <integer 0-100>,
    "notes": "<assessment of company details, website, contact info, registration>"
  }},
  "grammar_score": <integer 0-100 where 100=perfect grammar>,
  "salary_assessment": {{
    "is_realistic": <true|false>,
    "market_context": "<brief salary context for this type of role in India>"
  }},
  "recommended_actions": [
    "<action 1>",
    "<action 2>",
    "<action 3>"
  ],
  "scam_type": "<one of: Financial Scam | Data Harvesting | Pyramid Scheme | Advance Fee Fraud | Fake Employer | Legitimate Job | Unclear>",
  "confidence": <integer 0-100>
}}

Be thorough. Real scams often look polished. Focus on:
1. Upfront payment requests (immediate disqualifier)
2. Unrealistic salary vs. role requirements
3. Missing company verification details
4. Urgency tactics
5. Contact method legitimacy
6. Grammar and professional tone
7. Request for sensitive personal info early"""


def _build_comparison_prompt(jd_text: str, score: int) -> str:
    return f"""Based on this job description (scam score: {score}/100), generate a side-by-side educational comparison.

Original JD snippet (first 300 chars): {jd_text[:300]}

Return ONLY valid JSON:
{{
  "fake_markers": [
    "<specific element from the JD that is a fake marker with brief reason>"
  ],
  "real_equivalent": [
    "<what a legitimate job posting would say instead>"
  ],
  "key_differences": [
    {{
      "aspect": "<aspect name>",
      "scam_version": "<how scams present this>",
      "legitimate_version": "<how real jobs present this>"
    }}
  ],
  "rewrite_tip": "<one actionable tip on spotting this scam type>"
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _score_color(score: int) -> str:
    if score >= 75: return "#ef4444"
    if score >= 50: return "#f97316"
    if score >= 25: return "#eab308"
    return "#22c55e"


def _score_label(score: int) -> str:
    if score >= 75: return "⚠️ HIGH RISK"
    if score >= 50: return "🟠 SUSPICIOUS"
    if score >= 25: return "🟡 CAUTION"
    return "✅ LIKELY SAFE"


def _severity_class(sev: str) -> str:
    return {"HIGH": "", "MEDIUM": "warn", "LOW": "caution"}.get(sev, "")


def _render_score_meter(score: int, verdict: str, reason: str):
    color = _score_color(score)
    label = _score_label(score)
    st.markdown(f"""
<div class="scam-meter-wrap">
  <div style="display:flex; align-items:flex-end; gap:16px; flex-wrap:wrap;">
    <div>
      <div class="scam-score-number" style="color:{color}">{score}</div>
      <div class="scam-score-label" style="color:{color}">/ 100  Scam Risk</div>
    </div>
    <div style="flex:1; min-width:180px;">
      <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:#f1f5f9; margin-bottom:4px;">
        {verdict}
      </div>
      <div style="font-family:'DM Mono',monospace; font-size:12.5px; color:#94a3b8; line-height:1.5;">
        {reason}
      </div>
      <span class="verdict-badge" style="margin-top:10px; background:rgba({_hex_to_rgb(color)},0.12); color:{color}; border:1px solid rgba({_hex_to_rgb(color)},0.3);">
        {label}
      </span>
    </div>
  </div>
  <div class="scam-bar-track">
    <div class="scam-bar-fill" style="width:{score}%; background:linear-gradient(90deg, #22c55e, #eab308 40%, #f97316 70%, #ef4444);"></div>
  </div>
  <div style="display:flex; justify-content:space-between; font-family:'DM Mono',monospace; font-size:10.5px; color:#475569;">
    <span>0 — SAFE</span><span>25 — CAUTION</span><span>50 — SUSPICIOUS</span><span>75 — HIGH RISK</span><span>100 — SCAM</span>
  </div>
</div>
""", unsafe_allow_html=True)


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def _render_red_flags(flags: list):
    if not flags:
        st.markdown('<div class="flag-card safe"><div class="flag-title">✅ No significant red flags detected</div><div class="flag-desc">The description appears to follow standard job posting conventions.</div></div>', unsafe_allow_html=True)
        return
    for f in flags:
        sev = f.get("severity", "LOW")
        cls = _severity_class(sev)
        icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "💡"}.get(sev, "ℹ️")
        evidence = f.get("evidence", "")
        explanation = f.get("explanation", "")
        flag_name = f.get("flag", "Unknown Flag")
        st.markdown(f"""
<div class="flag-card {cls}">
  <div class="flag-title">{icon} [{sev}] {flag_name}</div>
  <div class="flag-desc">Evidence: "{evidence[:120]}{"..." if len(evidence)>120 else ""}"</div>
  <div class="flag-desc" style="margin-top:5px; opacity:0.9;">{explanation}</div>
</div>""", unsafe_allow_html=True)


def _render_comparison(comp_data: dict):
    fake_markers = comp_data.get("fake_markers", [])
    real_equiv = comp_data.get("real_equivalent", [])
    key_diffs = comp_data.get("key_differences", [])

    col1, col2 = st.columns(2)
    with col1:
        items = "".join(f"<li style='margin:5px 0;'>🚩 {m}</li>" for m in fake_markers[:6])
        st.markdown(f"""
<div class="compare-card fake">
  <h4>🚨 Scam Markers</h4>
  <ul style="padding-left:16px; margin:0; color:#fca5a5;">{items}</ul>
</div>""", unsafe_allow_html=True)

    with col2:
        items = "".join(f"<li style='margin:5px 0;'>✅ {m}</li>" for m in real_equiv[:6])
        st.markdown(f"""
<div class="compare-card real">
  <h4>✅ Legitimate Equivalent</h4>
  <ul style="padding-left:16px; margin:0; color:#86efac;">{items}</ul>
</div>""", unsafe_allow_html=True)

    if key_diffs:
        st.markdown('<div class="section-hdr">Key Differences</div>', unsafe_allow_html=True)
        for d in key_diffs[:5]:
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.markdown(f"<div style='font-family:Syne,sans-serif; font-size:12px; font-weight:700; color:#94a3b8; padding-top:8px;'>{d.get('aspect','')}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='flag-card' style='margin:4px 0;'><div class='flag-desc'>🚨 {d.get('scam_version','')}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='flag-card safe' style='margin:4px 0;'><div class='flag-desc'>✅ {d.get('legitimate_version','')}</div></div>", unsafe_allow_html=True)

    tip = comp_data.get("rewrite_tip", "")
    if tip:
        st.markdown(f'<div class="info-callout">💡 <b>Pro Tip:</b> {tip}</div>', unsafe_allow_html=True)


def _render_company_intel(company: dict, salary: dict, grammar_score: int):
    c1, c2, c3 = st.columns(3)
    co_score = company.get("score", 50)
    co_color = _score_color(100 - co_score)  # invert: higher legitimacy = greener

    with c1:
        st.markdown(f"""
<div style="text-align:center; padding:16px; background:rgba(30,41,59,0.6); border-radius:12px; border:1px solid rgba(148,163,184,0.1);">
  <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; color:{_score_color(100-co_score)};">{co_score}</div>
  <div style="font-family:'DM Mono',monospace; font-size:11px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Company Score</div>
  <div style="font-family:'DM Mono',monospace; font-size:11.5px; color:#cbd5e1; margin-top:8px;">{company.get('notes','')[:80]}</div>
</div>""", unsafe_allow_html=True)

    with c2:
        sal_real = salary.get("is_realistic", True)
        sal_color = "#22c55e" if sal_real else "#ef4444"
        sal_icon = "✅" if sal_real else "🚨"
        sal_label = "Realistic" if sal_real else "Unrealistic"
        st.markdown(f"""
<div style="text-align:center; padding:16px; background:rgba(30,41,59,0.6); border-radius:12px; border:1px solid rgba(148,163,184,0.1);">
  <div style="font-size:36px;">{sal_icon}</div>
  <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700; color:{sal_color}; margin-top:4px;">{sal_label}</div>
  <div style="font-family:'DM Mono',monospace; font-size:10.5px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase;">Salary</div>
  <div style="font-family:'DM Mono',monospace; font-size:11px; color:#cbd5e1; margin-top:8px;">{salary.get('market_context','')[:80]}</div>
</div>""", unsafe_allow_html=True)

    with c3:
        g_color = _score_color(100 - grammar_score)
        st.markdown(f"""
<div style="text-align:center; padding:16px; background:rgba(30,41,59,0.6); border-radius:12px; border:1px solid rgba(148,163,184,0.1);">
  <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; color:{_score_color(100-grammar_score)};">{grammar_score}</div>
  <div style="font-family:'DM Mono',monospace; font-size:11px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Grammar Score</div>
  <div style="font-family:'DM Mono',monospace; font-size:11.5px; color:#cbd5e1; margin-top:8px;">{"Professional quality" if grammar_score >= 75 else "Poor grammar detected — common in scams"}</div>
</div>""", unsafe_allow_html=True)


def _render_actions(actions: list, scam_type: str):
    type_colors = {
        "Financial Scam": "#ef4444",
        "Data Harvesting": "#f97316",
        "Pyramid Scheme": "#a855f7",
        "Advance Fee Fraud": "#ef4444",
        "Fake Employer": "#f97316",
        "Legitimate Job": "#22c55e",
        "Unclear": "#94a3b8",
    }
    color = type_colors.get(scam_type, "#94a3b8")
    st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
  <span style="font-family:'DM Mono',monospace; font-size:11px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase;">Scam Type:</span>
  <span style="font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:{color}; background:rgba({_hex_to_rgb(color)},0.1); padding:4px 14px; border-radius:99px; border:1px solid rgba({_hex_to_rgb(color)},0.25);">{scam_type}</span>
</div>""", unsafe_allow_html=True)

    for i, action in enumerate(actions, 1):
        st.markdown(f"""
<div style="display:flex; gap:12px; align-items:flex-start; padding:10px 0; border-bottom:1px solid rgba(148,163,184,0.08);">
  <span style="font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:#6366f1; min-width:24px;">{i:02d}</span>
  <span style="font-family:'DM Mono',monospace; font-size:12.5px; color:#cbd5e1;">{action}</span>
</div>""", unsafe_allow_html=True)


def _render_pattern_library():
    st.markdown('<div class="section-hdr">Known Scam Patterns — Reference Library</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (name, info) in enumerate(SCAM_PATTERNS.items()):
        with cols[i % 2]:
            examples_html = "".join(f"<li style='margin:3px 0;font-size:11px;color:#94a3b8;'>{e}</li>" for e in info["examples"])
            st.markdown(f"""
<div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.1); border-left:3px solid {info['color']}; border-radius:10px; padding:12px 14px; margin:6px 0;">
  <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:{info['color']}; margin-bottom:4px;">{info['icon']} {name}</div>
  <div style="font-family:'DM Mono',monospace; font-size:11.5px; color:#cbd5e1; margin-bottom:6px;">{info['desc']}</div>
  <ul style="padding-left:14px; margin:0;">{examples_html}</ul>
</div>""", unsafe_allow_html=True)


def _render_history(history: list):
    if not history:
        st.markdown('<div class="info-callout">No analyses yet this session. Run a detection above to see your history.</div>', unsafe_allow_html=True)
        return

    for item in reversed(history[-10:]):
        score = item.get("score", 0)
        color = _score_color(score)
        st.markdown(f"""
<div class="hist-item">
  <div style="flex:1;">
    <span style="font-family:'DM Mono',monospace; font-size:12px; color:#94a3b8;">{item.get('timestamp','')}</span>
    <span style="font-family:'DM Mono',monospace; font-size:12.5px; color:#cbd5e1; margin-left:12px;">{item.get('preview','')[:60]}…</span>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-family:'Syne',sans-serif; font-size:16px; font-weight:800; color:{color};">{score}</span>
    <span style="font-family:'DM Mono',monospace; font-size:11px; color:{color}; background:rgba({_hex_to_rgb(color)},0.1); padding:3px 10px; border-radius:99px;">{item.get('verdict','')}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TAB RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_job_scam_detector_tab(username: str):
    """
    Main entry point. Call this from your app's tab router:
        render_job_scam_detector_tab(st.session_state["username"])
    """
    _inject_css()

    # ── Session state init ─────────────────────────────────────────────────────
    if "jsd_history" not in st.session_state:
        st.session_state.jsd_history = []
    if "jsd_last_result" not in st.session_state:
        st.session_state.jsd_last_result = None
    if "jsd_last_comp" not in st.session_state:
        st.session_state.jsd_last_comp = None
    if "jsd_jd_text" not in st.session_state:
        st.session_state.jsd_jd_text = ""

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="tab-hero">
  <h1>🔍 Job Scam Detector</h1>
  <p>AI-powered detection of fake job descriptions, phishing listings, and financial scams</p>
</div>
""", unsafe_allow_html=True)

    # ── Rate-limit gate ────────────────────────────────────────────────────────
    allowed, gate_msg = check_and_gate_feature(username, FEATURE_KEY)

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    tab_detect, tab_compare, tab_library, tab_history = st.tabs([
        "🔬 Detect Scam",
        "⚖️ Fake vs. Real",
        "📚 Pattern Library",
        "🕓 History",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DETECT SCAM
    # ══════════════════════════════════════════════════════════════════════════
    with tab_detect:
        st.markdown('<div class="section-hdr">Paste Job Description</div>', unsafe_allow_html=True)

        col_input, col_quick = st.columns([3, 1])
        with col_quick:
            st.markdown("**Quick Load Examples**")
            if st.button("🚨 Load Fake JD", use_container_width=True):
                st.session_state.jsd_jd_text = EXAMPLE_FAKE_JD
                st.rerun()
            if st.button("✅ Load Real JD", use_container_width=True):
                st.session_state.jsd_jd_text = EXAMPLE_REAL_JD
                st.rerun()
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.jsd_jd_text = ""
                st.session_state.jsd_last_result = None
                st.session_state.jsd_last_comp = None
                st.rerun()

        with col_input:
            jd_text = st.text_area(
                "Job Description Text",
                value=st.session_state.jsd_jd_text,
                height=280,
                placeholder="Paste the full job description here. Include title, company, salary, duties, contact info — the more detail, the more accurate the detection.",
                label_visibility="collapsed",
                key="jsd_input_area",
            )
            st.session_state.jsd_jd_text = jd_text

        word_count = len(jd_text.split()) if jd_text.strip() else 0
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:11px;color:#475569;margin-top:-8px;">{word_count} words</div>', unsafe_allow_html=True)

        # ── Analyse button ─────────────────────────────────────────────────────
        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            analyse_clicked = st.button(
                "🔬 Analyse for Scam",
                use_container_width=True,
                type="primary",
                disabled=not allowed,
            )
        with col_info:
            if not allowed:
                st.markdown(gate_msg, unsafe_allow_html=True)
            else:
                _, gate_info = check_and_gate_feature(username, FEATURE_KEY)
                st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:11.5px;color:#475569;padding-top:10px;">⚡ {gate_info}</div>', unsafe_allow_html=True)

        # ── Run analysis ───────────────────────────────────────────────────────
        if analyse_clicked:
            if not jd_text.strip():
                st.warning("Please paste a job description to analyse.")
            elif word_count < 10:
                st.warning("Job description seems too short. Please add more detail for accurate detection.")
            else:
                with st.spinner("🤖 Scanning for scam signals…"):
                    try:
                        prompt = _build_analysis_prompt(jd_text)
                        raw = call_llm(prompt, st.session_state, model="llama-3.3-70b-versatile", temperature=0)

                        # ── Parse JSON ─────────────────────────────────────────
                        # Strip possible markdown fences
                        clean = raw.strip()
                        if clean.startswith("```"):
                            clean = re.sub(r"^```[a-z]*\n?", "", clean)
                            clean = re.sub(r"\n?```$", "", clean)
                        result = json.loads(clean)

                        st.session_state.jsd_last_result = result

                        # ── Comparison (secondary call, non-blocking) ──────────
                        score = result.get("scam_score", 50)
                        try:
                            comp_raw = call_llm(
                                _build_comparison_prompt(jd_text, score),
                                st.session_state,
                                model="llama-3.3-70b-versatile",
                                temperature=0,
                            )
                            comp_clean = comp_raw.strip()
                            if comp_clean.startswith("```"):
                                comp_clean = re.sub(r"^```[a-z]*\n?", "", comp_clean)
                                comp_clean = re.sub(r"\n?```$", "", comp_clean)
                            st.session_state.jsd_last_comp = json.loads(comp_clean)
                        except Exception:
                            st.session_state.jsd_last_comp = None

                        # ── Record usage + history ─────────────────────────────
                        record_feature_usage(username, FEATURE_KEY)
                        ist = pytz.timezone("Asia/Kolkata")
                        ts = datetime.now(ist).strftime("%d %b %Y, %I:%M %p")
                        st.session_state.jsd_history.append({
                            "timestamp": ts,
                            "preview": jd_text[:80],
                            "score": result.get("scam_score", 0),
                            "verdict": result.get("verdict", ""),
                        })

                    except json.JSONDecodeError:
                        st.error("Failed to parse AI response. Please try again.")
                        st.session_state.jsd_last_result = None
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        st.session_state.jsd_last_result = None

        # ── Display results ────────────────────────────────────────────────────
        result = st.session_state.jsd_last_result
        if result:
            st.divider()

            # Score meter
            _render_score_meter(
                result.get("scam_score", 0),
                result.get("verdict", ""),
                result.get("verdict_reason", ""),
            )

            # Intelligence panels
            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                st.markdown('<div class="section-hdr">🚩 Red Flags Detected</div>', unsafe_allow_html=True)
                _render_red_flags(result.get("red_flags", []))

                positives = result.get("positive_signals", [])
                if positives:
                    st.markdown('<div class="section-hdr">✅ Positive Signals</div>', unsafe_allow_html=True)
                    for p in positives:
                        st.markdown(f'<div class="flag-card safe"><div class="flag-desc">{p}</div></div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="section-hdr">📊 Intelligence Breakdown</div>', unsafe_allow_html=True)
                _render_company_intel(
                    result.get("company_legitimacy", {}),
                    result.get("salary_assessment", {}),
                    result.get("grammar_score", 50),
                )

                st.markdown('<div class="section-hdr" style="margin-top:20px;">🛡️ Recommended Actions</div>', unsafe_allow_html=True)
                _render_actions(
                    result.get("recommended_actions", []),
                    result.get("scam_type", "Unclear"),
                )

                # Confidence
                conf = result.get("confidence", 80)
                st.markdown(f"""
<div style="margin-top:12px; padding:10px 14px; background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.15); border-radius:8px;">
  <span style="font-family:'DM Mono',monospace; font-size:11px; color:#6366f1; letter-spacing:2px; text-transform:uppercase;">AI Confidence</span>
  <span style="font-family:'Syne',sans-serif; font-size:20px; font-weight:800; color:#a5b4fc; margin-left:12px;">{conf}%</span>
</div>""", unsafe_allow_html=True)

            # ── Export ──────────────────────────────────────────────────────────
            st.divider()
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                export_text = f"""HIRELYZER JOB SCAM ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
User: {username}
{'='*50}

SCAM SCORE: {result.get('scam_score', 0)}/100
VERDICT: {result.get('verdict', '')}
SUMMARY: {result.get('verdict_reason', '')}

SCAM TYPE: {result.get('scam_type', '')}
AI CONFIDENCE: {result.get('confidence', 0)}%

RED FLAGS ({len(result.get('red_flags', []))} detected):
{chr(10).join(f"  [{f.get('severity','')}] {f.get('flag','')}: {f.get('evidence','')[:80]}" for f in result.get('red_flags', []))}

POSITIVE SIGNALS:
{chr(10).join(f"  + {p}" for p in result.get('positive_signals', []))}

COMPANY LEGITIMACY: {result.get('company_legitimacy', {}).get('score', 0)}/100
{result.get('company_legitimacy', {}).get('notes', '')}

GRAMMAR QUALITY: {result.get('grammar_score', 0)}/100
SALARY REALISTIC: {result.get('salary_assessment', {}).get('is_realistic', 'Unknown')}
{result.get('salary_assessment', {}).get('market_context', '')}

RECOMMENDED ACTIONS:
{chr(10).join(f"  {i+1}. {a}" for i, a in enumerate(result.get('recommended_actions', [])))}

{'='*50}
HIRELYZER — Protecting your career journey
"""
                st.download_button(
                    "📄 Download Report (.txt)",
                    data=export_text,
                    file_name=f"scam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with exp_col2:
                json_export = json.dumps(result, indent=2)
                st.download_button(
                    "📦 Download Raw JSON",
                    data=json_export,
                    file_name=f"scam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — FAKE vs. REAL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tab_compare:
        st.markdown('<div class="section-hdr">Side-by-Side Comparison</div>', unsafe_allow_html=True)

        comp = st.session_state.jsd_last_comp
        result = st.session_state.jsd_last_result

        if not comp:
            st.markdown("""
<div class="info-callout">
  💡 Run a detection in the <b>Detect Scam</b> tab first — the comparison will automatically generate based on your analysed job description.
  <br><br>Or explore the built-in example comparison below:
</div>""", unsafe_allow_html=True)

            # Built-in comparison using example data
            st.markdown('<div class="section-hdr" style="margin-top:20px;">📖 Example: Fake vs. Real Job Description</div>', unsafe_allow_html=True)
            col_f, col_r = st.columns(2)
            with col_f:
                st.markdown(f"""
<div class="compare-card fake" style="height:auto;">
  <h4>🚨 FAKE JD Example</h4>
  <pre style="font-size:11px; color:#fca5a5; white-space:pre-wrap; font-family:'DM Mono',monospace;">{EXAMPLE_FAKE_JD}</pre>
</div>""", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"""
<div class="compare-card real" style="height:auto;">
  <h4>✅ REAL JD Example</h4>
  <pre style="font-size:11px; color:#86efac; white-space:pre-wrap; font-family:'DM Mono',monospace;">{EXAMPLE_REAL_JD}</pre>
</div>""", unsafe_allow_html=True)

            # Static difference table
            st.markdown('<div class="section-hdr" style="margin-top:24px;">Universal Markers to Compare</div>', unsafe_allow_html=True)
            markers = [
                ("Company Identity", "Anonymous / 'Leading MNC'", "Full name, CIN / registration number"),
                ("Salary Mention", "Impossibly high, no range", "Market-rate range with LPA / CTC split"),
                ("Required Skills", "None / 'Anyone can do it'", "Specific tools, years of experience"),
                ("Application Method", "WhatsApp / personal Gmail", "Official career portal or company email"),
                ("Payment Requested", "Yes — 'registration fee'", "Never — legitimate jobs pay you"),
                ("Contact Details", "Unknown person, phone only", "Named HR, official domain email"),
                ("Job Duties", "Vague: 'data entry / typing'", "Specific: API development, Agile, etc."),
                ("Urgency / FOMO", "'Only 5 seats! Apply NOW!'", "No pressure — open until filled"),
            ]
            for aspect, scam, legit in markers:
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    st.markdown(f"<div style='font-family:Syne,sans-serif;font-size:12px;font-weight:700;color:#64748b;padding-top:6px;'>{aspect}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='flag-card' style='margin:3px 0;'><div class='flag-desc'>🚨 {scam}</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='flag-card safe' style='margin:3px 0;'><div class='flag-desc'>✅ {legit}</div></div>", unsafe_allow_html=True)
        else:
            score = result.get("scam_score", 50) if result else 50
            verdict = result.get("verdict", "") if result else ""
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:12px;color:#94a3b8;margin-bottom:12px;">Analysis from your last scan — Score: <b style="color:{_score_color(score)};">{score}/100</b> | {verdict}</div>', unsafe_allow_html=True)
            _render_comparison(comp)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — PATTERN LIBRARY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_library:
        st.markdown("""
<div class="info-callout">
  📚 This library documents <b>10 known scam patterns</b> with real-world examples. Bookmark this page — it's your field guide to spotting fake job postings.
</div>""", unsafe_allow_html=True)

        _render_pattern_library()

        st.markdown('<div class="section-hdr" style="margin-top:28px;">🚨 Red Flag Quick-Reference</div>', unsafe_allow_html=True)
        for name, info in SCAM_PATTERNS.items():
            st.markdown(
                f'<span class="pattern-pill" style="background:rgba({_hex_to_rgb(info["color"])},0.1);color:{info["color"]};border:1px solid rgba({_hex_to_rgb(info["color"])},0.25);">{info["icon"]} {name}</span>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-hdr" style="margin-top:28px;">🛡️ General Safety Checklist</div>', unsafe_allow_html=True)
        checklist = [
            ("Never pay to apply", "Legitimate employers never ask for money upfront for equipment, training, or background checks."),
            ("Verify company exists", "Search the company on MCA21, LinkedIn, and their official website. Check CIN/GSTIN."),
            ("Match salary to role", "Compare offered salary against Glassdoor / AmbitionBox for the specific title and city."),
            ("Use official channels only", "Apply through company career pages, Naukri, or LinkedIn — never WhatsApp/Telegram."),
            ("Protect your Aadhar/PAN", "No legitimate employer needs these at the application stage."),
            ("Research the recruiter", "Verify the recruiter's LinkedIn profile, company email domain, and phone number."),
            ("Read reviews", "Check Glassdoor, AmbitionBox, and Google Reviews for the hiring company."),
            ("Trust your instincts", "If something feels off — unrealistic pay, rushed timeline, vague role — it probably is."),
        ]
        for title, desc in checklist:
            st.markdown(f"""
<div style="display:flex; gap:14px; padding:10px 0; border-bottom:1px solid rgba(148,163,184,0.07);">
  <span style="color:#22c55e; font-size:16px; flex-shrink:0;">✓</span>
  <div>
    <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:#f1f5f9;">{title}</div>
    <div style="font-family:'DM Mono',monospace; font-size:11.5px; color:#94a3b8; margin-top:3px;">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        history = st.session_state.jsd_history
        st.markdown('<div class="section-hdr">Session Analysis History</div>', unsafe_allow_html=True)

        if history:
            total = len(history)
            avg_score = int(sum(h["score"] for h in history) / total)
            high_risk = sum(1 for h in history if h["score"] >= 75)
            safe_count = sum(1 for h in history if h["score"] < 25)

            m1, m2, m3, m4 = st.columns(4)
            for col, label, val, color in [
                (m1, "Total Scanned", total, "#6366f1"),
                (m2, "Avg Risk Score", avg_score, _score_color(avg_score)),
                (m3, "High Risk Found", high_risk, "#ef4444"),
                (m4, "Safe Jobs", safe_count, "#22c55e"),
            ]:
                with col:
                    col.markdown(f"""
<div style="text-align:center; padding:14px; background:rgba(30,41,59,0.5); border-radius:10px; border:1px solid rgba(148,163,184,0.1);">
  <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:{color};">{val}</div>
  <div style="font-family:'DM Mono',monospace; font-size:10.5px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">{label}</div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-hdr" style="margin-top:20px;">Recent Analyses</div>', unsafe_allow_html=True)

        _render_history(history)

        if history:
            if st.button("🗑️ Clear History", use_container_width=False):
                st.session_state.jsd_history = []
                st.rerun()

        st.markdown("""
<div class="info-callout" style="margin-top:20px;">
  ℹ️ History is session-based and resets on page refresh. Results are not stored server-side for privacy.
</div>""", unsafe_allow_html=True)
