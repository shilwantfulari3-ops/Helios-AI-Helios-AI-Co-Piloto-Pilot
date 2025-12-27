import streamlit as st
import httpx
import asyncio
import json
import PyPDF2
import docx
import io
import time
import sqlite3
import pandas as pd
import re
import datetime
from typing import List, Dict, Any, Optional, Tuple
import base64

# --- 1. AI CONFIGURATION (Google Gemini) ---
# The `API_KEY` is where you paste your key from Google AI Studio
# --- 1. AI CONFIGURATION (Google Gemini) ---
# [cite_start]Use the stable model version and fetch the API key securely from Streamlit secrets [cite: 201]
# --- 1. AI CONFIGURATION (Google Gemini) ---
# Use the stable model version and fetch the API key securely from Streamlit secrets
# --- 1. AI CONFIGURATION (Google Gemini) ---
# Use the stable model version and fetch the API key securely from Streamlit secrets
# --- 1. AI CONFIGURATION (Google Gemini) ---
# Use the stable model version and fetch the API key securely from Streamlit secrets
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key="
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
DB_NAME = "helios.db"

if not API_KEY:
    st.error("Please set the GEMINI_API_KEY in your .streamlit/secrets.toml file or Streamlit Cloud settings.")
# --- 2. LOCAL PARSING ENGINE (V8) ---
EDUCATION_LEVELS = {
    "bachelor": "Bachelor's", "b.s.": "Bachelor's", "b.a.": "Bachelor's",
    "master": "Master's", "m.s.": "Master's", "m.a.": "Master's", "mba": "Master's",
    "phd": "PhD", "doctorate": "PhD"
}
SKILL_KEYWORDS = [
    "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue", "Node.js", "SQL", 
    "PostgreSQL", "MySQL", "MongoDB", "AWS", "Azure", "GCP", "Docker", "Kubernetes", 
    "Terraform", "Git", "Agile", "Scrum", "Jira", "Machine Learning", "PyTorch", "TensorFlow",
    "Data Analysis", "Tableau", "Power BI", "Leadership", "Team Management"
]

def local_parse_resume(text: str) -> Dict[str, Any]:
    """
    NEW V12: 100% local, instant, free parser.
    Now includes project and certification counts.
    """
    text_lower = text.lower()
    
    # 1. Extract Experience
    exp_years = 0.0
    matches = re.findall(r"(\d+)\s*\+?\s*(?:-|to)?\s*(\d+)?\s+years?[\s,.]?of[\s,.]?experience", text_lower)
    if matches:
        all_exp = [float(y) for x in matches for y in x if y]
        if all_exp:
            exp_years = max(all_exp)
    elif (matches := re.findall(r"(\d+)\s+years?", text_lower)):
        all_exp = [float(x) for x in matches]
        if all_exp:
            exp_years = max(all_exp)

    # 2. Extract Education
    edu_level = "None"
    edu_val = 0
    for key, val in EDUCATION_LEVELS.items():
        if key in text_lower:
            current_val = education_level_to_int(val)
            if current_val > edu_val:
                edu_level = val
                edu_val = current_val
                
    # 3. Extract Location
    location = "Not found"
    loc_matches = re.findall(r"\b([A-Za-z\s]+,\s*[A-Za-z]{2,3})\b", text)
    if loc_matches:
        location = loc_matches[0]
    
    # 4. Extract Skills
    skills = list(set([skill for skill in SKILL_KEYWORDS if skill.lower() in text_lower]))
    
    # --- NEW V12: Extract Project & Cert Counts ---
    project_count = len(re.findall(r"\bproject[s]?\b", text_lower))
    cert_count = len(re.findall(r"\bcertification[s]?\b|\bcertified\b", text_lower))
    
    return {
        "total_experience_years": exp_years,
        "education_level": edu_level,
        "location": location,
        "skills": skills,
        "project_count": project_count,
        "certification_count": cert_count
    }

# --- 3. DATABASE SETUP (V8) ---

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            parsed_json TEXT,
            culture_text TEXT,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            raw_text TEXT,
            local_parse_json TEXT,
            deep_dive_json TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS team_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            author TEXT DEFAULT 'Recruiter',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(candidate_id) REFERENCES candidates(id)
        )
        ''')
        conn.commit()

# --- 4. AI PROMPTS & SCHEMAS (NEW V11) ---

# --- STAGE 1: "LIGHT" PARSE (FOR RANKING & FILTERS) ---
JD_LIGHT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "job_title": {"type": "STRING"},
        "min_experience_years": {"type": "NUMBER"},
        "education_requirement": {"type": "STRING", "description": "e.g., 'Bachelor''s'"},
        "required_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "nice_to_have_skills": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["job_title", "min_experience_years", "education_requirement", "required_skills"]
}
JD_LIGHT_PROMPT = "Extract data from this JD using the provided schema. Focus only on these fields."

# --- STAGE 2: "DEEP DIVE" (FULL 10-CALL ANALYSIS) ---
DEEP_DIVE_PROMPTS = {
    # Call 1: Strengths, Weaknesses, Summary, Achievement Impact
    "summary_sw": {
        "prompt": "Analyze the candidate's strengths, weaknesses, and achievement impact *for this specific role*. Provide a 5-bullet summary.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "candidate_summary": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Exactly 5 bullet points summarizing the candidate."},
                "strengths": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Top 3-5 strengths for this specific role."},
                "weaknesses": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Top 3-5 weaknesses or gaps for this role."},
                "achievement_impact_analysis": {"type": "STRING", "description": "Analysis of the candidate's achievement impact (1-10 scale justification)."}
            }
        }
    },
    # Call 2: Core Scores (Fit & Culture)
    "fit_scores": {
        "prompt": "Provide a Role Fit Score (0-100) with justification and a Culture Fit Score (0-100) with justification.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "role_fit_score": {"type": "NUMBER", "description": "A score from 0-100 for overall role fit."},
                "role_fit_justification": {"type": "STRING", "description": "A 2-3 sentence justification for the score."},
                "culture_fit_score": {"type": "NUMBER", "description": "A 0-100 score for culture fit."},
                "culture_fit_reasoning": {"type": "STRING", "description": "Justification based on resume language vs. culture notes."}
            }
        }
    },
    # Call 3: Skills, ATS, and Match Map
    "skills_ats": {
        "prompt": "Analyze all skills. Provide key technical, soft, and hidden (inferred) skills. Then, provide an ATS score (0-100) and a Match Map of JD keywords.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "skill_analysis": {
                    "type": "OBJECT",
                    "properties": {
                        "key_technical_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "soft_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "hidden_skills": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Skills implied but not stated (e.t., 'leadership' from 'managed a team')."}
                    }
                },
                "ats_score": {"type": "NUMBER", "description": "A 0-100 keyword match score."},
                "match_map": {
                    "type": "OBJECT",
                    "properties": {
                        "exact_matches": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "partial_matches": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "missing_keywords": {"type": "ARRAY", "items": {"type": "STRING"}}
                    }
                }
            }
        }
    },
    # Call 4: Risk & Red Flags
    "red_flags": {
        "prompt": "Analyze the resume for red flags: employment gaps, job-hopping, unclear roles, or potential exaggerations/fake experience. Provide a boolean for each and reasoning. Also provide a 1-sentence AI Signal of overall resume quality.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "red_flags": {
                    "type": "OBJECT",
                    "properties": {
                        "has_employment_gaps": {"type": "BOOLEAN"},
                        "is_job_hopping": {"type": "BOOLEAN"},
                        "has_unclear_roles": {"type": "BOOLEAN"},
                        "has_potential_exaggerations": {"type": "BOOLEAN"},
                        "reasoning": {"type": "STRING", "description": "Justification if any flags are true."}
                    }
                },
                "ai_signal_reasoning": {"type": "STRING", "description": "1-sentence analysis of resume quality, clarity, and progression."}
            }
        }
    },
    # Call 5: Salary & ROI
    "salary_roi": {
        "prompt": "Estimate a salary range and recommended offer. Also predict job stability, training cost, and hiring ROI.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "salary_estimate": {
                    "type": "OBJECT",
                    "properties": {
                        "min": {"type": "NUMBER"}, "max": {"type": "NUMBER"}, "currency": {"type": "STRING", "description": "e.g., 'USD'"},
                        "justification": {"type": "STRING", "description": "Based on experience, location, and skills."}
                    }
                },
                "recommended_offer": {"type": "STRING", "description": "A specific recommended offer, e.g., '$135,000'"},
                "job_stability_prediction": {"type": "NUMBER", "description": "Score 0-100 of likelihood to stay 2+ years."},
                "training_cost_roi": {"type": "STRING", "description": "e.g., 'Low Training Cost, High ROI'"}
            }
        }
    },
    # Call 6: Growth & Resume Quality
    "growth_quality": {
        "prompt": "Provide resume improvement suggestions, a skill gap roadmap, and recommended job roles. Also score the resume's quality and writing.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "growth_plan": {
                    "type": "OBJECT",
                    "properties": {
                        "resume_improvements": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "skill_gap_roadmap": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "An upskilling plan for missing skills."}
                    }
                },
                "recommended_roles": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "3-5 other job roles this candidate fits."},
                "resume_quality_score": {"type": "NUMBER", "description": "0-100 score for design, content, and ATS-friendliness."},
                "writing_quality_analysis": {"type": "STRING", "description": "1-2 sentences on clarity, grammar, and professionalism."}
            }
        }
    },
    # Call 7: Interview Kit
    "interview_kit": {
        "prompt": "Generate 3 technical, 3 behavioral, and 2 situational interview questions.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "interview_questions": {
                    "type": "OBJECT",
                    "properties": {
                        "technical": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "behavioral": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "situational": {"type": "ARRAY", "items": {"type": "STRING"}}
                    }
                }
            }
        }
    },
    # Call 8: Communication Tools
    "comms": {
        "prompt": "Generate 3 recruiter email templates: 1. Interview Invitation, 2. Polite Rejection, 3. Request for More Information.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "communication_templates": {
                    "type": "OBJECT",
                    "properties": {
                        "interview_call": {"type": "STRING", "description": "Template for an interview invitation email."},
                        "rejection": {"type": "STRING", "description": "Template for a polite rejection email."},
                        "info_request": {"type": "STRING", "description": "Template to request more information."}
                    }
                }
            }
        }
    },
    # Call 9: AI Predictions (Personality)
    "predictions": {
        "prompt": "Predict the candidate's personality using the Big Five model (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) and their preferred work environment.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "personality_prediction": {"type": "STRING", "description": "Summary based on Big Five (e.g., 'High Conscientiousness, High Openness...')."},
                "preferred_environment": {"type": "STRING", "description": "e.g., 'Startup', 'Corporate', 'Remote', 'Hybrid'"}
            }
        }
    },
    # Call 10: AI Predictions (Career & LinkedIn)
    "forecast": {
        "prompt": "Analyze the candidate's career progression forecast (next 3-5 years). Also, analyze the provided LinkedIn URL (if any) for consistency and credibility.",
        "schema": {
            "type": "OBJECT",
            "properties": {
                "career_progression_forecast": {"type": "STRING", "description": "Their likely next role or career path in 3-5 years."},
                "linkedin_analysis": {"type": "STRING", "description": "Analysis of the LinkedIn URL. If no URL, state that. Check for consistency with the resume."}
            }
        }
    },
    # Call 11: Comparison Prompt (Text-gen)
    "compare": {
        "prompt": """
        You are the lead AI Recruitment Advisor. You will be given a job's requirements,
        the company's culture, and two candidates (A and B).
        
        Your task is to generate a detailed, side-by-side comparison in Markdown
        to help me decide who to hire.
        
        1.  Create a comparison table with rows:
            - **Role Fit** (A brief summary)
            - **Strengths** (Key advantages for this role)
            - **Weaknesses** (Potential gaps or concerns)
            - **Role Fit Score** (The 0-100 score)
            - **ATS Score**
            - **Job Stability Prediction**
            - **Culture Alignment**
            - **Red Flags**
            - **Salary Estimate**
        2.  After the table, provide a final "Gemini's Recommendation" and
            justify your choice in 2-3 sentences.
        
        ---
        CONTEXT:
        Job Data: {job_data}
        Company Culture: {company_culture}
        Candidate A (Full Deep Dive Analysis): {candidate_a_data}
        Candidate B (Full Deep Dive Analysis): {candidate_b_data}
        ---
        
        Generate the Candidate Comparison now.
        """,
        "schema": None 
    },
    # --- NEW V12: Pipeline Chat Prompt ---
    "pipeline_chat": {
        "prompt": """
        You are an AI Recruitment Analyst. You will be given a JSON object containing a
        list of all candidates in the pipeline. This data is from a *fast local parser*
        and includes *estimated* counts for projects and certifications.

        Your task is to answer the user's question *based only on this data*.
        Be concise and factual. If the user asks who has the "most projects,"
        find the candidate with the highest 'project_count' and state their name
        and the count.

        ---
        CONTEXT (Candidate Pipeline Data):
        {pipeline_data}
        ---
        USER QUESTION:
        {user_question}
        ---

        Generate the answer now.
        """,
        "schema": None
    }
}


# --- 5. ASYNC API & DB CALL FUNCTIONS ---

async def call_gemini_api(payload: Dict[str, Any], is_chat: bool = False) -> Dict[str, Any]:
    """Helper function to call the Gemini API with retry logic."""
    headers = {'Content-Type': 'application/json'}
    url = f"{API_URL}{API_KEY}"
    
    if not API_KEY:
        raise ValueError("API_KEY is not set. Please add your Google AI API key at the top of the script.")

    async with httpx.AsyncClient(timeout=180.0) as client:
        retries = 3
        delay = 2
        for attempt in range(retries):
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if not result.get("candidates") or not result["candidates"][0].get("content"):
                    raise Exception("Invalid AI response format.")

                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                if is_chat:
                    return {"text": response_text}
                else:
                    return json.loads(response_text)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    st.error("ERROR: 403 Forbidden. Your API key is likely invalid or has not been enabled. Please check your Google AI Studio settings.")
                    raise
                print(f"API call error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
                else:
                    raise
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}. AI response was: {response_text}")
                raise Exception(f"AI returned invalid JSON. Response: {response_text}")
            except Exception as e:
                print(f"Non-HTTP error: {e}")
                raise

async def run_ai_schema_parse(system_prompt: str, schema: Dict[str, Any], context_text: str) -> Dict[str, Any]:
    """Runs the AI parsing with a specific schema."""
    payload = {
        "contents": [{"parts": [{"text": context_text}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }
    return await call_gemini_api(payload)

async def run_ai_copilot(prompt: str) -> str:
    """Runs the co-pilot generative prompts."""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    result = await call_gemini_api(payload, is_chat=True)
    return result["text"]

# --- 6. FILE READING FUNCTIONS ---

def read_pdf(file_stream: io.BytesIO) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}"); return ""

def read_docx(file_stream: io.BytesIO) -> str:
    try:
        document = docx.Document(file_stream)
        text = ""
        for para in document.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}"); return ""

def read_csv(file_stream: io.BytesIO) -> pd.DataFrame:
    try:
        return pd.read_csv(file_stream)
    except Exception as e:
        st.error(f"Error reading CSV: {e}"); return pd.DataFrame()

# --- 7. CORE APP LOGIC (SCORING & FILTERS) ---

def education_level_to_int(level_str: str) -> int:
    level = (level_str or "none").lower()
    if "phd" in level: return 3
    if "master" in level: return 2
    if "bachelor" in level: return 1
    return 0

def calculate_local_score(job_data: Dict[str, Any], cand_data: Dict[str, Any]) -> int:
    """Calculates the "local" score for ranking, based on local parse data."""
    
    # 1. Required Skills (60%)
    job_req_skills = set([s.lower() for s in job_data.get('required_skills', [])])
    cand_skills = set([s.lower() for s in cand_data.get('skills', [])])
    matched_req_skills = job_req_skills.intersection(cand_skills)
    skill_match_score = len(matched_req_skills) / len(job_req_skills) if job_req_skills else 1.0
    
    # 2. Experience (20%)
    job_exp = job_data.get('min_experience_years', 0)
    cand_exp = cand_data.get('total_experience_years', 0)
    exp_match_score = min(cand_exp / job_exp, 1.0) if job_exp > 0 else 1.0
        
    # 3. Education (10%)
    job_edu_req = education_level_to_int(job_data.get('education_requirement', 'None'))
    cand_edu_level = education_level_to_int(cand_data.get('education_level', 'None'))
    education_score = 1.0 if cand_edu_level >= job_edu_req else 0.0
    
    # 4. Proxy/Bonus (10%)
    proxy_score = min(len(cand_skills) / 20.0, 1.0) # Simple proxy
    
    total_score = (
        (skill_match_score * 0.60) +
        (exp_match_score * 0.20) +
        (education_score * 0.10) +
        (proxy_score * 0.10)
    )
    return int(total_score * 100)

def filter_candidates(scored_candidates, filters):
    filtered_list = scored_candidates
    
    min_exp, max_exp = filters['experience']
    filtered_list = [
        (c_id, data, score) for c_id, data, score in filtered_list
        if min_exp <= data.get('total_experience_years', 0) <= max_exp
    ]
    
    degrees = filters['degree']
    if degrees:
        degrees_int = [education_level_to_int(d) for d in degrees]
        min_degree_req = min(degrees_int) if degrees_int else 0
        filtered_list = [
            (c_id, data, score) for c_id, data, score in filtered_list
            if education_level_to_int(data.get('education_level', 'None')) >= min_degree_req
        ]
        
    location_query = filters['location'].lower()
    if location_query:
        filtered_list = [
            (c_id, data, score) for c_id, data, score in filtered_list
            if location_query in data.get('location', '').lower()
        ]
    return filtered_list

# --- 8. STREAMLIT APP (NEW V10 UI) ---

def initialize_session_state():
    if 'active_job_id' not in st.session_state:
        st.session_state.active_job_id = None
    if 'active_job_title' not in st.session_state:
        st.session_state.active_job_title = "No Job Selected"
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"

# --- NEW V10: Custom CSS & PWA ---
def setup_page_config():
    """Sets page config, loads CSS, and injects PWA manifest."""
    
    st.set_page_config(
        page_title="Helios v12",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # --- PWA & App Icon Setup ---
    # Create an SVG icon (simple rocket emoji)
    icon_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="192" height="192" viewBox="0 0 100 100">
        <text x="50" y="50" font-size="80" dominant-baseline="central" text-anchor="middle">🚀</text>
    </svg>
    """
    # Encode it for the manifest
    icon_svg_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(icon_svg.encode('utf-8')).decode('utf-8')}"

    # Create the manifest.json content
    manifest = {
        "name": "Helios AI Co-Pilot",
        "short_name": "Helios",
        "icons": [
            {
                "src": icon_svg_data_uri,
                "sizes": "192x192",
                "type": "image/svg+xml"
            }
        ],
        "theme_color": "#007BFF",
        "background_color": "#F8F9FA",
        "start_url": ".",
        "display": "standalone", # This makes it open like an app
        "scope": "/"
    }
    manifest_json = json.dumps(manifest)
    manifest_b64 = base64.b64encode(manifest_json.encode('utf-8')).decode('utf-8')
    manifest_href = f"data:application/json;base64,{manifest_b64}"

    st.markdown(f"""
    <!-- PWA / App Manifest -->
    <meta name="theme-color" content="#007BFF">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="Helios">
    <link rel="apple-touch-icon" href="{icon_svg_data_uri}">
    <link rel="manifest" href="{manifest_href}">

    <!-- Custom CSS Styles -->
    <style>
        /* --- Global --- */
        .stApp {{
            background-color: #F8F9FA; /* Light gray background */
        }}
        
        /* --- Sidebar --- */
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid #E0E0E0;
            padding: 1rem;
        }}
        
        /* --- Main Content --- */
        .main [data-testid="stBlock"] {{
            padding-top: 2rem;
        }}
        
        /* --- NEW V10: Sidebar Navigation Buttons --- */
        [data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            text-align: left;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: none;
            font-size: 1.05rem;
            font-weight: 600;
            color: #4F4F4F; /* Dark gray text */
            transition: all 0.3s ease;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: #F0F8FF; /* Light blue hover */
            color: #007BFF;
        }}
        /* Active page button style */
        [data-testid="stSidebar"] .stButton > button.active-page {{
            background-color: #E6F2FF; /* Primary light */
            color: #007BFF; /* Primary */
        }}

        /* --- Containers & Expanders --- */
        [data-testid="stExpander"], [data-testid="stContainer"][border="true"] {{
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            background-color: #FFFFFF;
        }}
        [data-testid="stExpander"] > summary {{
            font-weight: 600;
        }}
        
        /* --- Metrics --- */
        [data-testid="stMetric"] {{
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            padding: 1rem;
        }}
        [data-testid="stMetric"] > label {{
            font-weight: 600;
            color: #4F4F4F;
        }}
        [data-testid="stMetric"] > div[data-testid="stMetricValue"] {{
            font-size: 2.25rem;
            color: #007BFF;
        }}
        
        /* --- Tabs --- */
        .stTabs [data-baseweb="tab"] {{
            font-weight: 600;
            font-size: 1rem;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: #007BFF;
            border-bottom-color: #007BFF;
        }}
        
        /* --- Primary Button --- */
        .stButton > button[kind="primary"] {{
            background-color: #007BFF;
            color: white;
            border: none;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: #0056b3;
            color: white;
        }}
        
        /* --- NEW V10: Deep Dive At-a-Glance Card --- */
        .at-a-glance-card {{
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Async Runner Functions (V8) ---
async def run_job_parser(jd_text, culture_text):
    with st.spinner("🤖 Calling AI to analyze job description..."):
        try:
            parsed_json = await run_ai_schema_parse(JD_LIGHT_PROMPT, JD_LIGHT_SCHEMA, jd_text)
            parsed_json_str = json.dumps(parsed_json)
            
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO jobs (title, parsed_json, culture_text, raw_text) VALUES (?, ?, ?, ?)",
                    (parsed_json['job_title'], parsed_json_str, culture_text, jd_text)
                )
                conn.commit()
                st.session_state.active_job_id = cursor.lastrowid
                st.session_state.active_job_title = parsed_json['job_title']
            st.success(f"Job '{parsed_json['job_title']}' saved and activated!")
        except Exception as e:
            st.error(f"Failed to parse job description: {e}")

async def run_local_parsers(files_to_parse: List[Tuple[str, str]]):
    if not st.session_state.active_job_id:
        st.error("No active job selected. Please set a job in Tab 2."); return
    
    job_id = st.session_state.active_job_id
    progress_bar = st.progress(0, text="Starting local parse...")
    
    candidates_to_insert = []
    
    # --- NEW V12: 100% Local Parse ---
    for i, (filename, text) in enumerate(files_to_parse):
        progress_text = f"Locally parsing {filename}... ({i+1}/{len(files_to_parse)})"
        progress_bar.progress((i + 1) / len(files_to_parse), text=progress_text)
        try:
            # Step 1: Run fast local parse
            local_parse_data = local_parse_resume(text)
            local_parse_json = json.dumps(local_parse_data)
            candidates_to_insert.append((job_id, filename, text, local_parse_json))
        except Exception as e:
            st.warning(f"Failed to parse {filename}: {e}")
            
    progress_bar.progress(1.0, text="Saving results to database...")
    with get_db() as conn:
        conn.executemany(
            "INSERT INTO candidates (job_id, filename, raw_text, local_parse_json) VALUES (?, ?, ?, ?)",
            candidates_to_insert
        )
        conn.commit()
    
    progress_bar.empty()
    st.success(f"Successfully parsed and saved {len(candidates_to_insert)} resumes!")

async def run_deep_dive(candidate_id: int, job_data: Dict, culture: str, resume_text: str, linkedin_url: str):
    """
    NEW V11: Runs the full 10-call Deep Dive for a single candidate.
    """
    with st.spinner(f"🤖 Performing Deep Dive analysis... (This may take a minute)"):
        try:
            full_report = {}
            # Base context for all calls
            base_context_text = f"Job: {json.dumps(job_data)}\nCulture: {culture}\nResume: {resume_text}"
            
            # Context for LinkedIn call
            linkedin_context_text = f"Resume: {resume_text}\nLinkedIn URL: {linkedin_url}"

            # Create all 10 parallel tasks
            tasks = []
            for key, config in DEEP_DIVE_PROMPTS.items():
                if key in ["compare", "pipeline_chat"]: continue # Skip compare/chat prompts
                
                # Use specific context for the 'forecast' (LinkedIn) call
                context = linkedin_context_text if key == "forecast" else base_context_text
                
                if config['schema']:
                    tasks.append(run_ai_schema_parse(config['prompt'], config['schema'], context))
            
            results = await asyncio.gather(*tasks)
            
            # Combine all results into one flat report
            for result in results:
                full_report.update(result)
            
            # Save the complete report to DB
            with get_db() as conn:
                conn.execute(
                    "UPDATE candidates SET deep_dive_json = ? WHERE id = ?",
                    (json.dumps(full_report), candidate_id)
                )
                conn.commit()
            st.success(f"Deep Dive complete and saved!")
        except Exception as e:
            st.error(f"Failed to run Deep Dive: {e}")

# --- NEW V10: Page Rendering Functions ---
def render_dashboard():
    st.header("Dashboard")
    if not st.session_state.active_job_id:
        st.info("No job selected. Please select a job from the sidebar or create one in 'Job Setup'.")
    else:
        with get_db() as conn:
            job_data = conn.execute("SELECT * FROM jobs WHERE id = ?", (st.session_state.active_job_id,)).fetchone()
        
        if job_data:
            job_id = st.session_state.active_job_id
            with get_db() as conn:
                total_cands = conn.execute("SELECT COUNT(*) FROM candidates WHERE job_id = ?", (job_id,)).fetchone()[0]
                deep_dives_run = conn.execute("SELECT COUNT(*) FROM candidates WHERE job_id = ? AND deep_dive_json IS NOT NULL", (job_id,)).fetchone()[0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Active Job", st.session_state.active_job_title)
            c2.metric("Total Candidates in Pipeline", total_cands)
            c3.metric("Candidates Awaiting Deep Dive", total_cands - deep_dives_run)
            
            st.divider()
            st.subheader("Your Hiring Funnel")
            st.markdown(f"""
            1.  **Job Setup:** You have an active job. To create a new one, go to 'Job Setup'.
            2.  **Candidate Pipeline:** Go to 'Candidate Pipeline' to upload resumes for **{st.session_state.active_job_title}**.
            3.  **Filter & Rank:** Use the filters in the Pipeline to find your top candidates.
            4.  **Deep Dive & Compare:** Go to 'Deep Dive & Compare' to run the full AI analysis.
            5.  **AI Pipeline Chat:** Go to 'AI Pipeline Chat' to ask questions about your entire candidate pool.
            """)
        else:
            st.info("No job data found for this ID.")

def render_job_setup():
    st.header("1. Job Setup")
    
    with st.container(border=True):
        st.subheader("Create New Job Profile")
        st.markdown("Create a new job profile. This will be saved to the database.")
        with st.form("new_job_form"):
            jd_text = st.text_area("Job Description", height=250, placeholder="Paste your job description here...")
            culture_text = st.text_area("Company Culture", height=200, placeholder="Describe your company culture...")
            submitted = st.form_submit_button("Create and Activate Job Profile", type="primary", use_container_width=True)
            
            if submitted:
                if jd_text and culture_text:
                    asyncio.run(run_job_parser(jd_text, culture_text))
                    st.rerun()
                else:
                    st.warning("Please fill out *both* the Job Description and Company Culture.")
    
    # --- NEW V11: Save/Load History ---
    with st.container(border=True):
        st.subheader("Save & Load Analysis")
        st.markdown("Save your current job and all its candidates to a JSON file, or load a previous session.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.active_job_id:
                # Create the full session data
                with get_db() as conn:
                    job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (st.session_state.active_job_id,)).fetchone())
                    cands = [dict(r) for r in conn.execute("SELECT * FROM candidates WHERE job_id = ?", (st.session_state.active_job_id,)).fetchall()]
                
                full_session_data = {
                    "job": job,
                    "candidates": cands
                }
                st.download_button(
                    label="Save Analysis Session",
                    data=json.dumps(full_session_data, indent=2, default=str), # Add default=str for datetimes
                    file_name=f"helios_analysis_{st.session_state.active_job_title.replace(' ','_')}_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
        with col2:
            loaded_file = st.file_uploader("Load Analysis Session (.json)", type="json")
            if loaded_file is not None:
                try:
                    loaded_data = json.load(loaded_file)
                    job_data = loaded_data.get("job", {})
                    candidates_data = loaded_data.get("candidates", [])
                    
                    with get_db() as conn:
                        # Insert job, handling potential None values
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO jobs (title, parsed_json, culture_text, raw_text, created_at) VALUES (?, ?, ?, ?, ?)",
                            (job_data.get('title'), job_data.get('parsed_json'), job_data.get('culture_text'), job_data.get('raw_text'), job_data.get('created_at'))
                        )
                        new_job_id = cursor.lastrowid
                        
                        # Insert candidates
                        for cand in candidates_data:
                            conn.execute(
                                """INSERT INTO candidates (job_id, filename, raw_text, local_parse_json, deep_dive_json) 
                                   VALUES (?, ?, ?, ?, ?)""",
                                (new_job_id, cand.get('filename'), cand.get('raw_text'), cand.get('local_parse_json'), cand.get('deep_dive_json'))
                            )
                        conn.commit()
                        
                    st.session_state.active_job_id = new_job_id
                    st.session_state.active_job_title = job_data.get('title')
                    st.success(f"Successfully loaded job '{job_data.get('title')}' with {len(candidates_data)} candidates!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load file: {e}")

def render_pipeline():
    st.header("2. Candidate Pipeline")
    if not st.session_state.active_job_id:
        st.warning("Please select or create a job first in '1. Job Setup'."); st.stop()

    job_id = st.session_state.active_job_id

    with st.expander("Upload Candidates"):
        with st.container(border=True):
            st.subheader("Upload from ATS (CSV)")
            ats_file = st.file_uploader("Upload CSV", type="csv")
            if st.button("Parse ATS CSV", use_container_width=True):
                if ats_file:
                    df = read_csv(ats_file)
                    if "resume_text" in df.columns:
                        files_to_parse = []
                        for index, row in df.iterrows():
                            filename = f"csv_row_{index}"
                            if 'name' in df.columns: filename = row['name']
                            files_to_parse.append((filename, row['resume_text']))
                        asyncio.run(run_local_parsers(files_to_parse))
                    else:
                        st.error("CSV must contain a 'resume_text' column.")
                else:
                    st.warning("Please upload a CSV file.")
        
        with st.container(border=True):
            st.subheader("Upload Resumes (PDF/DOCX) & LinkedIn")
            uploaded_files = st.file_uploader(
                "Choose resumes (PDF or DOCX)...",
                type=["pdf", "docx"],
                accept_multiple_files=True
            )
            if st.button("Parse Uploaded Resumes", type="primary", use_container_width=True):
                if uploaded_files:
                    files_to_parse = []
                    for file in uploaded_files:
                        file_stream = io.BytesIO(file.getvalue())
                        text = ""
                        if file.name.endswith(".pdf"): text = read_pdf(file_stream)
                        elif file.name.endswith(".docx"): text = read_docx(file_stream)
                        if text:
                            files_to_parse.append((file.name, text))
                    asyncio.run(run_local_parsers(files_to_parse))
                else:
                    st.warning("Please upload at least one resume.")
            
            st.text_input("LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/...", key="linkedin_url_pipeline")
            st.caption("How to parse from LinkedIn: Go to the profile > More > Save to PDF, then upload the PDF here.")
    
    st.divider()

    with get_db() as conn:
        job_data_row = conn.execute("SELECT parsed_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        candidates = conn.execute("SELECT id, filename, local_parse_json FROM candidates WHERE job_id = ?", (job_id,)).fetchall()
    
    if not job_data_row: st.warning("Job data not found."); st.stop()
    if not candidates: st.info("No candidates uploaded for this job yet."); st.stop()

    job_data = json.loads(job_data_row['parsed_json'])
    
    scored_candidates = []
    for cand in candidates:
        try:
            cand_data = json.loads(cand['local_parse_json'])
            cand_data['filename'] = cand['filename']
            score = calculate_local_score(job_data, cand_data)
            scored_candidates.append((cand['id'], cand_data, score))
        except Exception as e:
            st.error(f"Error scoring {cand['filename']}: {e}")
    
    st.subheader("Filters")
    with st.container(border=True):
        all_exp = [float(c.get('total_experience_years', 0)) for _, c, _ in scored_candidates]
        min_exp_val, max_exp_val = (0.0, 20.0)
        if all_exp:
            min_exp_val, max_exp_val = (min(all_exp), max(all_exp) + 1.0)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            exp_filter = st.slider("Experience (Years)", 0.0, 30.0, (min_exp_val, max_exp_val), 0.5)
        with c2:
            degree_filter = st.multiselect("Minimum Degree", ["Bachelor's", "Master's", "PhD"])
        with c3:
            location_filter = st.text_input("Location (contains...)")
    
    filters = {"experience": exp_filter, "degree": degree_filter, "location": location_filter}
    filtered_list = filter_candidates(scored_candidates, filters)
    
    st.subheader(f"Ranked Candidates (Showing {len(filtered_list)} of {len(scored_candidates)})")
    filtered_list.sort(key=lambda x: x[2], reverse=True)
    
    # --- Polished Candidate Card (NEW V12: Added Proj/Cert counts) ---
    for i, (cand_id, cand_data, score) in enumerate(filtered_list):
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col1:
                st.metric(label=f"Rank #{i+1}", value=f"{score}%")
            with col2:
                st.subheader(cand_data.get('filename', 'N/A'))
                st.caption(f"Exp: {cand_data.get('total_experience_years', 0)} yrs | Edu: {cand_data.get('education_level', 'N/A')} | Loc: {cand_data.get('location', 'N/A')}")
            with col3:
                st.markdown("**Local Parse Estimates**")
                st.caption(f"Projects: {cand_data.get('project_count', 0)} | Certs: {cand_data.get('certification_count', 0)}")
            with col4:
                st.write("") # Spacer
                if st.button("Select for Deep Dive", key=f"select_{cand_id}", use_container_width=True):
                    st.session_state.selected_candidate_id = cand_id
                    st.success(f"Selected. Go to 'Deep Dive & Compare' page.")
                    st.session_state.page = "Deep Dive"
                    st.rerun()

def render_deep_dive_compare():
    st.header("3. Deep Dive & Compare")
    if not st.session_state.active_job_id:
        st.warning("Please select or create a job first in '1. Job Setup'."); st.stop()
    
    job_id = st.session_state.active_job_id
    
    with get_db() as conn:
        job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        candidates_rows = conn.execute("SELECT id, filename, local_parse_json, deep_dive_json, raw_text FROM candidates WHERE job_id = ?", (job_id,)).fetchall()
    
    if not candidates_rows:
        st.info("No candidates in pipeline. Please upload resumes in 'Candidate Pipeline'."); st.stop()
        
    candidate_options = {row['filename']: row['id'] for row in candidates_rows}
    candidate_options_list = [r['filename'] for r in candidates_rows]
    
    # --- Deep Dive Section ---
    st.subheader("Candidate Deep Dive (Full Analysis)")
    
    search_term = st.text_input("Search for a candidate by filename...", "")
    if search_term:
        filtered_options_list = [name for name in candidate_options_list if search_term.lower() in name.lower()]
    else:
        filtered_options_list = candidate_options_list
    
    default_index = 0
    if 'selected_candidate_id' in st.session_state and st.session_state.selected_candidate_id in [r['id'] for r in candidates_rows]:
        try:
            selected_filename = [r['filename'] for r in candidates_rows if r['id'] == st.session_state.selected_candidate_id][0]
            if selected_filename in filtered_options_list:
                default_index = filtered_options_list.index(selected_filename)
        except Exception: pass
    
    if not filtered_options_list:
        st.warning(f"No candidates found matching '{search_term}'."); st.stop()
        
    selected_filename = st.selectbox(
        "Select a candidate to analyze:", 
        filtered_options_list,
        index=default_index,
        key="deep_dive_select"
    )
    selected_cand_id = candidate_options.get(selected_filename)
    
    if not selected_cand_id:
        st.info("Select a candidate to begin."); st.stop()
        
    with get_db() as conn:
        cand_row = conn.execute("SELECT * FROM candidates WHERE id = ?", (selected_cand_id,)).fetchone()
    
    cand_deep_dive = json.loads(cand_row['deep_dive_json']) if cand_row['deep_dive_json'] else None
    
    linkedin_url = st.text_input("Add LinkedIn URL for analysis (optional)", placeholder="https://www.linkedin.com/in/...", key=f"linkedin_deepdive_{selected_cand_id}")
    
    if not cand_deep_dive:
        if st.button(f"Run Deep Dive Analysis for {cand_row['filename']}", type="primary", use_container_width=True):
            job_data = json.loads(job_row['parsed_json'])
            culture_text = job_row['culture_text']
            resume_text = cand_row['raw_text']
            asyncio.run(run_deep_dive(selected_cand_id, job_data, culture_text, resume_text, linkedin_url))
            st.rerun()
    else:
        # --- RENDER THE 16-POINT REPORT (V11 UI - BUG FIXES) ---
        report = cand_deep_dive
        st.title(f"Analysis for: {cand_row['filename']}")
        
        col1, col2 = st.columns([1, 1.5]) # Main layout columns
        
        with col1:
            # --- Left "At-a-Glance" Card ---
            with st.container():
                st.markdown("<div class='at-a-glance-card'>", unsafe_allow_html=True)
                st.subheader("At-a-Glance Summary")
                
                c1, c2 = st.columns(2)
                c1.metric("Overall Role Fit Score", f"{report.get('role_fit_score', 0)}%")
                c2.metric("Culture Fit Score", f"{report.get('culture_fit_score', 0)}%")
                st.caption(report.get('role_fit_justification', '...'))

                st.subheader("Salary Estimate")
                salary_est = report.get('salary_estimate', {})
                st.success(f"**{salary_est.get('min', 0):,} - {salary_est.get('max', 0):,} {salary_est.get('currency', 'N/A')}**")
                st.caption(f"Recruiter Offer: {report.get('recommended_offer', 'N/A')}")
                
                st.subheader("Key Predictions")
                st.markdown(f"**Stability:** {report.get('job_stability_prediction', 0)}% chance to stay 2+ years")
                st.markdown(f"**Training/ROI:** {report.get('training_cost_roi', '...')}")
                st.markdown(f"**Personality:** {report.get('personality_prediction', '...')}")
                
                st.subheader("Red Flags")
                red_flags_data = report.get('red_flags', {})
                red_flag_bools = [v for k, v in red_flags_data.items() if k.startswith('has_') and v]
                if any(red_flag_bools):
                    st.error(f"**Flags Detected:** {red_flags_data.get('reasoning', 'See flags below.')}", icon="🚩")
                else:
                    st.success("**No Red Flags Detected**", icon="✅")
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # --- Right "Details" Tabs ---
            tab_names = ["S/W & Summary", "Skill Match Map", "Interview & Comms", "Growth & Quality", "Team Notes", "Raw Data"]
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs(tab_names)
            
            with sub_tab1:
                with st.container(border=True):
                    st.subheader("Candidate Summary")
                    for item in report.get('candidate_summary', []): st.markdown(f"- {item}")
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Strengths")
                        st.multiselect("Strengths", report.get('strengths', []), report.get('strengths', []), disabled=True, key=f"str_{selected_cand_id}")
                    with c2:
                        st.subheader("Weaknesses")
                        st.multiselect("Weaknesses", report.get('weaknesses', []), report.get('weaknesses', []), disabled=True, key=f"weak_{selected_cand_id}")
                with st.container(border=True):
                    st.subheader("Achievement Impact")
                    st.info(report.get('achievement_impact_analysis', '...'))

            with sub_tab2:
                with st.container(border=True):
                    st.subheader("JD Match Map & ATS Score")
                    ats_data = report.get('skills_ats', {})
                    st.metric("Keyword ATS Score", f"{ats_data.get('ats_score', 0)}%")
                    
                    match_data = ats_data.get('match_map', {})
                    exact = len(match_data.get('exact_matches', []))
                    partial = len(match_data.get('partial_matches', []))
                    missing = len(match_data.get('missing_keywords', []))
                    total_skills = exact + partial + missing
                    
                    if total_skills > 0:
                        st.markdown("**Skill Match Breakdown**")
                        chart_data = pd.DataFrame(
                            {
                                "count": [exact, partial, missing],
                                "color": ["#28A745", "#FFC107", "#DC3545"] # Add color column
                            },
                            index=["Exact Matches", "Partial Matches", "Missing Keywords"] # Use row index
                        )
                        st.bar_chart(chart_data, y="count", color="color")

                    with st.expander("Show Keyword & Skill Lists"):
                        c1, c2, c3 = st.columns(3)
                        c1.multiselect("Exact", match_data.get('exact_matches', []), match_data.get('exact_matches', []), disabled=True, key=f"ex_{selected_cand_id}")
                        c2.multiselect("Partial", match_data.get('partial_matches', []), match_data.get('partial_matches', []), disabled=True, key=f"par_{selected_cand_id}")
                        c3.multiselect("Missing", match_data.get('missing_keywords', []), match_data.get('missing_keywords', []), disabled=True, key=f"miss_{selected_cand_id}")
            
            with sub_tab3:
                with st.container(border=True):
                    st.subheader("Tailored Interview Kit")
                    interview_data = report.get('interview_questions', {})
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Behavioral & Situational**")
                        for q in interview_data.get('behavioral', []): st.markdown(f"- {q}")
                        for q in interview_data.get('situational', []): st.markdown(f"- {q}")
                    with c2:
                        st.markdown("**Technical Questions**")
                        for q in interview_data.get('technical', []): st.markdown(f"- {q}")
                
                with st.container(border=True):
                    st.subheader("Communication Tools")
                    comms = report.get('communication_templates', {})
                    with st.expander("Show Interview Email"):
                        st.code(comms.get('interview_call', '...'), language='text')
                    with st.expander("Show Rejection Email"):
                        st.code(comms.get('rejection', '...'), language='text')
                    with st.expander("Show Info Request Email"):
                        st.code(comms.get('info_request', '...'), language='text')

            with sub_tab4:
                with st.container(border=True):
                    st.subheader("Candidate Growth Plan")
                    growth_data = report.get('growth_plan', {})
                    st.markdown("**Resume Improvement Suggestions**")
                    for imp in growth_data.get('resume_improvements', []): st.markdown(f"- {imp}")
                    st.markdown("**Skill Gap Upskilling Roadmap**")
                    for road in growth_data.get('skill_gap_roadmap', []): st.markdown(f"- {road}")
                
                with st.container(border=True):
                    st.subheader("Other AI Insights")
                    st.metric("Resume Quality Score", f"{report.get('resume_quality_score', 0)}%")
                    st.markdown(f"**Writing Quality:** {report.get('writing_quality_analysis', '...')}")
                    st.markdown(f"**Recommended Roles:** {', '.join(report.get('recommended_roles', []))}")
                    st.markdown(f"**Career Forecast:** {report.get('career_progression_forecast', '...')}")
                    st.markdown(f"**LinkedIn Analysis:** {report.get('linkedin_analysis', '...')}")
            
            with sub_tab5:
                with st.container(border=True):
                    st.subheader("Collaborative Team Notes")
                    with get_db() as conn:
                        notes = conn.execute(
                            "SELECT * FROM team_notes WHERE candidate_id = ? ORDER BY created_at DESC",
                            (selected_cand_id,)
                        ).fetchall()
                    
                    for note in notes:
                        st.info(f"**{note['author']}** ({note['created_at']}):\n\n{note['note']}")
                    
                    with st.form("new_note_form"):
                        note_text = st.text_area("Add a new note...")
                        if st.form_submit_button("Save Note", use_container_width=True):
                            if note_text:
                                with get_db() as conn:
                                    conn.execute(
                                        "INSERT INTO team_notes (candidate_id, note) VALUES (?, ?)",
                                        (selected_cand_id, note_text)
                                    )
                                    conn.commit()
                                st.rerun()
            
            with sub_tab6:
                with st.container(border=True):
                    st.subheader("Full Raw JSON from AI")
                    st.json(report)
    
    st.divider()
    # --- Candidate Compare Section ---
    with st.container(border=True):
        st.header("Candidate Compare")
        st.markdown("Select any two candidates who have had a **Deep Dive** analysis completed.")
        
        deep_dive_candidates = [r for r in candidates_rows if r['deep_dive_json']]
        
        if len(deep_dive_candidates) < 2:
            st.warning("Please run a 'Deep Dive' analysis on at least two candidates to compare.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                cand_a_id = st.selectbox(
                    "Select Candidate A", 
                    [r['id'] for r in deep_dive_candidates], 
                    format_func=lambda x: [r['filename'] for r in deep_dive_candidates if r['id'] == x][0],
                    key="comp_a"
                )
            with c2:
                cand_b_id = st.selectbox(
                    "Select Candidate B",
                    [r['id'] for r in deep_dive_candidates if r['id'] != cand_a_id],
                    format_func=lambda x: [r['filename'] for r in deep_dive_candidates if r['id'] == x][0],
                    key="comp_b"
                )
            
            if st.button("Compare Candidates", type="primary", use_container_width=True):
                if cand_a_id and cand_b_id:
                    with st.spinner("🤖 Generating AI comparison..."):
                        cand_a_data = json.loads([r['deep_dive_json'] for r in deep_dive_candidates if r['id'] == cand_a_id][0])
                        cand_b_data = json.loads([r['deep_dive_json'] for r in deep_dive_candidates if r['id'] == cand_b_id][0])
                        
                        cand_a_data['filename'] = [r['filename'] for r in deep_dive_candidates if r['id'] == cand_a_id][0]
                        cand_b_data['filename'] = [r['filename'] for r in deep_dive_candidates if r['id'] == cand_b_id][0]
                        
                        context = {
                            "job_data": json.dumps(json.loads(job_row['parsed_json'])),
                            "company_culture": job_row['culture_text'],
                            "candidate_a_data": json.dumps(cand_a_data),
                            "candidate_b_data": json.dumps(cand_b_data),
                        }
                        try:
                            compare_md = asyncio.run(run_ai_copilot(DEEP_DIVE_PROMPTS['compare']['prompt'].format(**context)))
                            st.markdown(compare_md)
                        except Exception as e:
                            st.error(f"Could not generate comparison: {e}")
                else:
                    st.error("Please select two different candidates.")

# --- NEW V12: AI Pipeline Chat Page ---
def render_pipeline_chat():
    st.header("📊 AI Pipeline Chat")
    st.markdown("Ask questions about your *entire* candidate pipeline for the active job.")
    
    if not st.session_state.active_job_id:
        st.warning("Please select or create a job first in '1. Job Setup'."); st.stop()
    
    job_id = st.session_state.active_job_id
    with get_db() as conn:
        job_data_row = conn.execute("SELECT parsed_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        candidates = conn.execute("SELECT id, filename, local_parse_json FROM candidates WHERE job_id = ?", (job_id,)).fetchall()

    if not candidates:
        st.info("No candidates in pipeline. Please upload resumes in 'Candidate Pipeline'."); st.stop()
        
    # --- Build the context for the AI ---
    pipeline_data = []
    for cand in candidates:
        data = json.loads(cand['local_parse_json'])
        pipeline_data.append({
            "filename": cand['filename'],
            "experience_years": data.get('total_experience_years', 0),
            "education": data.get('education_level', 'N/A'),
            "location": data.get('location', 'N/A'),
            "project_count": data.get('project_count', 0), # From our new local parser
            "certification_count": data.get('certification_count', 0) # From our new local parser
        })

    st.info(f"AI is ready to analyze all **{len(pipeline_data)}** candidates in your pipeline.")

    user_question = st.text_input("Ask a question about the pipeline...", placeholder="e.g., Who has the most projects? or 'List all candidates with 5+ years experience'")
    
    if st.button("Ask AI", type="primary", use_container_width=True):
        if user_question:
            with st.spinner("🤖 Analyzing the entire pipeline..."):
                context = {
                    "pipeline_data": json.dumps(pipeline_data),
                    "user_question": user_question
                }
                try:
                    response_md = asyncio.run(run_ai_copilot(DEEP_DIVE_PROMPTS['pipeline_chat']['prompt'].format(**context)))
                    st.markdown(response_md)
                except Exception as e:
                    st.error(f"Could not get response: {e}")
        else:
            st.warning("Please enter a question.")
            
# --- Main entry point ---
def main():
    # --- NEW V10: Must be first command ---
    setup_page_config()
    
    initialize_session_state() 
    init_db() 
    
    st.title("🚀 Helios v12 - The Final Demo")

    if not API_KEY:
        st.error("Hold on! You need to add your Google AI API key to the 'API_KEY' variable at the top of the script.", icon="🚨")
        return

    # --- NEW V10: Sidebar Navigation ---
    with st.sidebar:
        st.header("Active Job")
        with get_db() as conn:
            jobs = conn.execute("SELECT id, title FROM jobs ORDER BY created_at DESC").fetchall()
        
        if jobs:
            job_options = {job['title']: job['id'] for job in jobs}
            
            default_index = 0
            if st.session_state.active_job_id:
                try:
                    active_title = [title for title, id in job_options.items() if id == st.session_state.active_job_id][0]
                    default_index = list(job_options.keys()).index(active_title)
                except Exception: pass
            
            selected_title = st.selectbox(
                "Select a job to work on:", 
                job_options.keys(), 
                index=default_index,
                key="job_selector"
            )
            if selected_title:
                st.session_state.active_job_id = job_options[selected_title]
                st.session_state.active_job_title = selected_title
        else:
            st.session_state.active_job_id = None
            st.session_state.active_job_title = "No Job Selected"

        st.info(f"**Active Job:** {st.session_state.active_job_title}")
        st.markdown("---")
        
        st.header("Navigation")
        
        # This logic makes the active button highlighted
        page = st.session_state.page
        if st.button("🚀 Dashboard", use_container_width=True, type="secondary"):
            st.session_state.page = "Dashboard"; st.rerun()
        if st.button("📄 Job Setup", use_container_width=True, type="secondary"):
            st.session_state.page = "Job Setup"; st.rerun()
        if st.button("👥 Candidate Pipeline", use_container_width=True, type="secondary"):
            st.session_state.page = "Pipeline"; st.rerun()
        if st.button("🔍 Deep Dive & Compare", use_container_width=True, type="secondary"):
            st.session_state.page = "Deep Dive"; st.rerun()
        # --- NEW V12: AI Pipeline Chat Button ---
        if st.button("📊 AI Pipeline Chat", use_container_width=True, type="secondary"):
            st.session_state.page = "Pipeline Chat"; st.rerun()
            
        # Add custom class to active button
        st.markdown(f"""
        <script>
            var buttons = window.parent.document.querySelectorAll('[data-testid="stSidebar"] .stButton > button');
            buttons.forEach(function(btn) {{
                if (btn.innerText.includes("{page}")) {{
                    btn.classList.add('active-page');
                }} else {{
                    btn.classList.remove('active-page');
                }}
            }});
        </script>
        """, unsafe_allow_html=True)

    # --- NEW V10: Page Rendering Logic ---
    if st.session_state.page == "Dashboard":
        render_dashboard()
    elif st.session_state.page == "Job Setup":
        render_job_setup()
    elif st.session_state.page == "Pipeline":
        render_pipeline()
    elif st.session_state.page == "Deep Dive":
        render_deep_dive_compare()
    # --- NEW V12: AI Pipeline Chat Page ---
    elif st.session_state.page == "Pipeline Chat":
        render_pipeline_chat()


if __name__ == "__main__":
    main()