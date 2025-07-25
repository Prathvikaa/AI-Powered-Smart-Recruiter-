# AI-Powered-Smart-Recruiter-
A Tkinter-based desktop application that helps recruiters analyze candidate fit based on a job description, resume, and chat conversation transcript. It combines AI-powered analysis using LLaMA-3 (via Groq API) with resume-job description similarity scoring and generates a detailed candidate evaluation report.

**Features**

✅ Resume & Job Description Matching
      Automatically calculates similarity score between candidate resume and job description using Sentence-BERT embeddings

✅ Interactive Recruiter-Candidate Chat Interface
      Recruiter and candidate can exchange messages
      All chat history is stored and analyzed
      
✅ AI-Powered Candidate Analysis
      Uses LLaMA 3 (via Groq API) to analyze:
      Key technical & soft skills from the conversation
      Resume vs. conversation consistency
      Overall job fit analysis
      Final recommendation: Proceed / Decline / Additional Screening
      Confidence level with justification

✅ Visual Highlights in Analysis
      Important terms like PROCEED TO INTERVIEW, DECLINE, Key Insights are highlighted in color

✅ Resume Parsing & JD Loading
      Supports both PDF & TXT files for resumes and job descriptions

✅ Save Candidate Report
      Export full candidate evaluation report as a text file with metadata

**Tech Stack**

1) GUI: Tkinter
2) AI Model: LLaMA 3 (via Groq API)
3) Embeddings: Sentence-BERT (all-MiniLM-L6-v2)
4) Similarity Scoring: Cosine similarity (0-10 scale)
5) PDF Parsing: pdfplumber
6) Language Processing: re, datetime
7) Conversation Analysis: LangChain Groq

**How It Works**

1) Load Job Description & Candidate Resume
    The app extracts text from PDF or TXT files
    Calculates an initial resume-job description match score

2) Chat with Candidate
    Simulate a recruiter-candidate conversation
    Messages are timestamped and stored

3) AI-Powered Analysis
    Once the conversation is sufficient, click Analyze Conversation
    The system sends job description, resume & conversation transcript to the AI model
    AI provides a detailed candidate fit report

4) Save Report
    Save the full evaluation as a .txt file with metadata


