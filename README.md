Helios v12 - AI Recruitment Co-Pilot
Helios v12 is an automated recruitment platform that uses Google Gemini 1.5 Flash to transform resume screening into actionable AI intelligence. It evaluates candidates based on technical skill match, cultural alignment, and career trajectory.

🛠️ Key Features

Hybrid Parsing Engine: Uses a fast local regex-based engine for initial ranking and an 11-call AI "Deep Dive" for comprehensive candidate analysis.

AI-Driven Metrics: Generates ATS scores, job stability predictions, and red flag detection for employment gaps or exaggerations.

Recruiter Intelligence: Includes an AI Pipeline Chat to query your candidate pool in natural language.

Interview & Comms Kit: Automatically generates tailored technical questions and recruiter email templates.

🧰 Tech Stack

Frontend: Streamlit (with Custom CSS/PWA support).
AI Orchestration: Google Gemini API (Async Implementation).
Database: SQLite3 (Relational storage for Jobs, Candidates, and Notes).
Data Processing: PyPDF2, python-docx, and Pandas.

🚀 Getting Started
1. Installation
Install the required dependencies:
Bash
pip install -r requirements.txt

2. Configuration (Secure Setup)
Helios uses Streamlit Secrets for API management to ensure your keys are never public.
Create a folder named .streamlit in your project root.
Inside, create a file named secrets.toml.
Add your Google API Key:
Ini, TOML
GEMINI_API_KEY = "your_google_ai_studio_key_here"

3. Launching the App
On Windows: Double-click run_helios_app.bat.
Via Terminal: streamlit run helios_app.py.

📂 Repository Structure

helios_app.py: Core application logic and AI orchestration.
requirements.txt: Environment dependencies.
.gitignore: Prevents private secrets and databases from being uploaded.
run_helios_app.bat: Windows startup script.
