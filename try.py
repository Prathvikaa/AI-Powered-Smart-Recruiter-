import os
import pdfplumber
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
import datetime
import re
import reportlab.lib.pagesizes
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

#for multiple resume
# Initialize services with environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ API Key not found! Set the environment variable 'GROQ_API_KEY'.")

# Initialize models
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables
conversation_messages = []
job_description = ""
candidate_resumes = []
interview_suggestions = []  # New list to store dynamically generated suggestions


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + "\n"
    return text.strip()


def calculate_match_score(jd, resume):
    """Calculate similarity score between job description and resume"""
    jd_embedding = embedding_model.encode(jd, convert_to_tensor=True)
    resume_embedding = embedding_model.encode(resume, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
    return similarity * 10  # Convert to 0-10 scale


def generate_dynamic_interview_suggestion():
    """Generate an interview suggestion based on recent conversation messages"""
    global conversation_messages, job_description, candidate_resumes

    if len(conversation_messages) < 2 or not job_description or not candidate_resumes:
        return None

    # Use the last few messages for context
    recent_messages = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in conversation_messages[-3:]])

    prompt = f"""
    Based on the job description for a Data Analyst role and the recent conversation, 
    generate a targeted follow-up interview question or discussion point.

    Job Description Context:
    {job_description}

    Recent Conversation:
    {recent_messages}

    Provide:
    A question that is better to be asked for the role , no justification needed
    """

    try:
        response = llm.invoke(prompt)
        suggestion = re.sub(r"\\(.?)\\*", r"\1", response.content).strip()
        return suggestion
    except Exception as e:
        print(f"Error generating interview suggestion: {e}")
        return None


def generate_interview_questions(jd, resume):
    """Generate targeted interview questions based on job description and resume"""
    prompt = f"""
    Generate a list of 5-7 targeted interview questions based on the following:

    JOB DESCRIPTION:
    {jd}

    CANDIDATE RESUME:
    {resume}

    For each question, provide:
    1. A specific question targeting the candidate's skills or experience
    Format the output as a numbered list,just the questions enough no justification needed.
    """

    response = llm.invoke(prompt)
    return re.sub(r"\\(.?)\\*", r"\1", response.content).strip()


def analyze_candidate_fit():
    """Analyze if candidate is a good fit based on job description, resumes and conversation"""
    conversation_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in conversation_messages])

    # Combine multiple resumes if available
    combined_resumes = "\n\n---\n\n".join(candidate_resumes)

    prompt = f"""
    Based on the following information, provide a detailed analysis of whether this candidate should be advanced to the next interview stage:

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE RESUME(S):
    {combined_resumes}

    CONVERSATION TRANSCRIPT:
    {conversation_text}

    Please analyze and provide:
    1. Key Insights from Conversation (technical skills, communication)
    2. Job Fit Analysis
    3. FINAL RECOMMENDATION: PROCEED TO INTERVIEW / DECLINE / ADDITIONAL SCREENING
    4. Justification for recommendation

    Provide a short,structured, clear, and professional analysis.
    """

    response = llm.invoke(prompt)
    return re.sub(r"\\(.?)\\*", r"\1", response.content).strip()


def save_pdf_report(filename, analysis):
    """Save analysis to a PDF without asterisks"""
    doc = SimpleDocTemplate(filename, pagesize=reportlab.lib.pagesizes.letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Candidate Evaluation Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Date
    story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Remove markdown asterisks and create paragraphs
    clean_analysis = analysis.replace('*', '')
    sections = clean_analysis.split('\n\n')
    for section in sections:
        story.append(Paragraph(section, styles['Normal']))
        story.append(Spacer(1, 6))

    doc.build(story)


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recruiter-Candidate Chat Interface")
        self.root.geometry("900x800")  # Slightly taller to accommodate suggestions
        self.current_sender = tk.StringVar(value="Recruiter")

        # Initialize resume tracking
        self.candidate_resumes = []

        # Create UI components
        self.setup_ui()
        self.prompt_load_files()

    def setup_ui(self):
        """Set up the user interface components"""
        # Header frame
        header = tk.Frame(self.root, pady=10, padx=20)
        header.pack(fill=tk.X)

        self.jd_btn = tk.Button(header, text="Load Job Description", command=self.load_job_description)
        self.jd_btn.pack(side=tk.LEFT, padx=5)

        # Resume section
        resume_frame = tk.Frame(header)
        resume_frame.pack(side=tk.LEFT, padx=5)

        self.resume_btn = tk.Button(resume_frame, text="Load Candidate Resume", command=self.load_candidate_resume)
        self.resume_btn.pack(side=tk.TOP)

        self.resume_listbox = tk.Listbox(resume_frame, height=3, width=20)
        self.resume_listbox.pack(side=tk.TOP, pady=5)

        self.status_label = tk.Label(header, text="Please load job description and resume")
        self.status_label.pack(side=tk.LEFT, padx=20)

        self.analyze_btn = tk.Button(header, text="Analyze Conversation",
                                     command=self.analyze_conversation, bg="green", fg="white")
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)

        self.clear_btn = tk.Button(header, text="Clear Chat", command=self.clear_chat)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)

        # Interview Suggestion section
        suggestion_frame = tk.Frame(self.root, pady=5, padx=20)
        suggestion_frame.pack(fill=tk.X)

        self.suggestion_label = tk.Label(suggestion_frame, text="Interview Suggestion:", font=("Arial", 10, "bold"))
        self.suggestion_label.pack(side=tk.LEFT)

        self.suggestion_text = tk.Label(suggestion_frame, text="", wraplength=800, justify=tk.LEFT)
        self.suggestion_text.pack(side=tk.LEFT, expand=True)

        # Chat display
        chat_frame = tk.Frame(self.root, pady=10, padx=20)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, bg="white")
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Message input
        input_frame = tk.Frame(self.root, pady=10, padx=20)
        input_frame.pack(fill=tk.X)

        sender_frame = tk.Frame(input_frame)
        sender_frame.pack(fill=tk.X, pady=5)

        tk.Radiobutton(sender_frame, text="Recruiter", variable=self.current_sender,
                       value="Recruiter").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(sender_frame, text="Candidate", variable=self.current_sender,
                       value="Candidate").pack(side=tk.LEFT, padx=5)

        self.message_input = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.message_input.pack(fill=tk.X, pady=5)
        self.message_input.bind("<Control-Return>", lambda e: self.send_message())

        send_btn = tk.Button(input_frame, text="Send Message", command=self.send_message,
                             bg="blue", fg="white")
        send_btn.pack(side=tk.RIGHT, padx=5)

    def prompt_load_files(self):
        """Initial prompt to load files"""
        if messagebox.askyesno("Load Files", "Would you like to load job description and resume now?"):
            self.load_job_description()
            self.load_candidate_resume()

    def load_job_description(self):
        """Load job description from file"""
        global job_description
        file_path = filedialog.askopenfilename(title="Select Job Description",
                                               filetypes=[("PDF Files", ".pdf"), ("Text Files", ".txt")])
        if not file_path:
            return

        job_description = (extract_text_from_pdf(file_path) if file_path.endswith('.pdf')
                           else open(file_path, 'r', encoding='utf-8').read())

        self.jd_btn.config(bg="green", fg="white")
        self.update_status()

    def load_candidate_resume(self):
        """Load candidate resume(s) from file"""
        global candidate_resumes
        file_path = filedialog.askopenfilename(title="Select Candidate Resume",
                                               filetypes=[("PDF Files", ".pdf"), ("Text Files", ".txt")])
        if not file_path:
            return

        resume_text = (extract_text_from_pdf(file_path) if file_path.endswith('.pdf')
                       else open(file_path, 'r', encoding='utf-8').read())

        candidate_resumes.append(resume_text)
        self.resume_listbox.insert(tk.END, os.path.basename(file_path))
        self.update_status()

    def update_status(self):
        """Update status label based on loaded files"""
        global job_description, candidate_resumes

        if job_description and candidate_resumes:
            # Calculate match score with the first resume if multiple are loaded
            score = calculate_match_score(job_description, candidate_resumes[0])
            self.status_label.config(text=f"Initial Resume Match: {score:.1f}/10", fg="blue")
        elif job_description:
            self.status_label.config(text="Job description loaded. Please load resume.")
        elif candidate_resumes:
            self.status_label.config(text="Resume loaded. Please load job description.")
        else:
            self.status_label.config(text="Please load job description and resume")

    def send_message(self):
        """Send a message to the chat"""
        global conversation_messages, interview_suggestions

        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return

        sender = self.current_sender.get()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        conversation_messages.append({
            "sender": sender,
            "message": message,
            "timestamp": timestamp
        })

        # Display message
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, f"{sender}: ", f"{sender.lower()}")
        self.chat_display.insert(tk.END, f"{message}\n\n")

        # Apply tags/colors
        self.chat_display.tag_config("timestamp", foreground="gray")
        self.chat_display.tag_config("recruiter", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("candidate", foreground="green", font=("Arial", 10, "bold"))

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

        # Generate interview suggestion if conversation has some depth
        if len(conversation_messages) > 3:
            suggestion = generate_dynamic_interview_suggestion()
            if suggestion:
                interview_suggestions.append(suggestion)
                self.suggestion_text.config(text=suggestion)
            else:
                self.suggestion_text.config(text="")

        # Clear input and switch sender
        self.message_input.delete("1.0", tk.END)
        self.current_sender.set("Candidate" if sender == "Recruiter" else "Recruiter")

    def analyze_conversation(self):
        """Enhanced conversation analysis with interview questions"""
        global job_description, candidate_resumes

        # Validation
        if not job_description or not candidate_resumes:
            messagebox.showerror("Error", "Please load job description and at least one resume before analyzing")
            return

        if len(conversation_messages) < 3:
            messagebox.showwarning("Warning", "The conversation is too short. Add more messages before analyzing.")
            return

        # Display
        self.status_label.config(text="Analyzing conversation... Please wait.", fg="orange")
        self.root.update()

        # Get analysis and interview questions
        analysis = analyze_candidate_fit()
        interview_questions = generate_interview_questions(job_description, "\n\n".join(candidate_resumes))

        self.show_analysis_results(analysis, interview_questions)

    def show_analysis_results(self, analysis, interview_questions):
        """Show analysis results in a new window with save PDF option"""
        self.status_label.config(text="Analysis complete", fg="green")

        results_window = tk.Toplevel(self.root)
        results_window.title("Candidate Analysis Results")
        results_window.geometry("700x600")

        results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD,
                                                 padx=15, pady=15, font=("Arial", 11))
        results_text.pack(fill=tk.BOTH, expand=True)

        # Remove asterisks from text
        clean_analysis = analysis.replace('*', '')
        clean_questions = interview_questions.replace('*', '')

        # Display Analysis and Interview Questions
        results_text.insert(tk.END, "CANDIDATE ANALYSIS:\n\n")
        results_text.insert(tk.END, clean_analysis)
        results_text.insert(tk.END, "\n\n--- INTERVIEW QUESTIONS ---\n\n")
        results_text.insert(tk.END, clean_questions)

        # buttons
        button_frame = tk.Frame(results_window, pady=10)
        button_frame.pack()

        def save_report():
            file_path = filedialog.asksaveasfilename(
                title="Save Report",
                defaultextension=".pdf",
                filetypes=[("PDF Files", ".pdf")]
            )
            if file_path:
                save_pdf_report(file_path, analysis)
                messagebox.showinfo("Success", "Report saved successfully!")

        tk.Button(button_frame, text="Save PDF Report", command=save_report).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Close", command=results_window.destroy).pack(side=tk.LEFT, padx=10)

    def clear_chat(self):
        """Clear the chat history"""
        global conversation_messages, candidate_resumes, interview_suggestions

        if messagebox.askyesno("Confirm", "This will clear the entire conversation and loaded resumes. Continue?"):
            conversation_messages = []
            candidate_resumes.clear()
            interview_suggestions.clear()

            # Clear chat display
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)

            # Clear resume listbox
            self.resume_listbox.delete(0, tk.END)

            # Clear interview suggestion
            self.suggestion_text.config(text="")

            # Reset sender
            self.current_sender.set("Recruiter")

            # Reset status
            self.status_label.config(text="Please load job description and resume")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()