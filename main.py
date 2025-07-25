import os
import pdfplumber
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
import datetime
import re

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
candidate_resume = ""


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


def analyze_candidate_fit():
    """Analyze if candidate is a good fit based on job description, resume and conversation"""
    conversation_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in conversation_messages])

    prompt = f"""
    Based on the following information, provide a detailed analysis of whether this candidate should be advanced to the next interview stage:

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE RESUME:
    {candidate_resume}

    CONVERSATION TRANSCRIPT:
    {conversation_text}

    Please analyze and provide:
    1. Key Insights from Conversation (technical skills, communication, cultural fit, red flags)
    2. Resume vs. Conversation Consistency
    3. Job Fit Analysis
    4. FINAL RECOMMENDATION: PROCEED TO INTERVIEW / DECLINE / ADDITIONAL SCREENING
    5. Confidence Level: [High/Medium/Low]
    6. Justification for recommendation
    """

    response = llm.invoke(prompt)
    return re.sub(r"\\(.?)\\*", r"\1", response.content).strip()

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recruiter-Candidate Chat Interface")
        self.root.geometry("900x700")
        self.current_sender = tk.StringVar(value="Recruiter")

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

        self.resume_btn = tk.Button(header, text="Load Candidate Resume", command=self.load_candidate_resume)
        self.resume_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(header, text="Please load job description and resume")
        self.status_label.pack(side=tk.LEFT, padx=20)

        self.analyze_btn = tk.Button(header, text="Analyze Conversation",
                                     command=self.analyze_conversation, bg="green", fg="white")
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)

        self.clear_btn = tk.Button(header, text="Clear Chat", command=self.clear_chat)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)

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
        """Load candidate resume from file"""
        global candidate_resume
        file_path = filedialog.askopenfilename(title="Select Candidate Resume",
                                               filetypes=[("PDF Files", ".pdf"), ("Text Files", ".txt")])
        if not file_path:
            return

        candidate_resume = (extract_text_from_pdf(file_path) if file_path.endswith('.pdf')
                            else open(file_path, 'r', encoding='utf-8').read())

        self.resume_btn.config(bg="green", fg="white")
        self.update_status()

    def update_status(self):
        """Update status label based on loaded files"""
        global job_description, candidate_resume

        if job_description and candidate_resume:
            score = calculate_match_score(job_description, candidate_resume)
            self.status_label.config(text=f"Initial Resume Match: {score:.1f}/10", fg="blue")
        elif job_description:
            self.status_label.config(text="Job description loaded. Please load resume.")
        elif candidate_resume:
            self.status_label.config(text="Resume loaded. Please load job description.")
        else:
            self.status_label.config(text="Please load job description and resume")

    def send_message(self):
        """Send a message to the chat"""
        global conversation_messages

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

        # Clear input and switch sender
        self.message_input.delete("1.0", tk.END)
        self.current_sender.set("Candidate" if sender == "Recruiter" else "Recruiter")

    def analyze_conversation(self):
        """Analyze the conversation and provide recommendations"""
        global job_description, candidate_resume

        # Validation
        if not job_description or not candidate_resume:
            messagebox.showerror("Error", "Please load both job description and resume before analyzing")
            return

        if len(conversation_messages) < 3:
            messagebox.showwarning("Warning", "The conversation is too short. Add more messages before analyzing.")
            return

        # Display
        self.status_label.config(text="Analyzing conversation... Please wait.", fg="orange")
        self.root.update()

        analysis = analyze_candidate_fit()
        self.show_analysis_results(analysis)

    def show_analysis_results(self, analysis):
        """Show analysis results in a new window"""
        self.status_label.config(text="Analysis complete", fg="green")

        results_window = tk.Toplevel(self.root)
        results_window.title("Candidate Analysis Results")
        results_window.geometry("700x600")

        results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD,
                                                 padx=15, pady=15, font=("Arial", 11))
        results_text.pack(fill=tk.BOTH, expand=True)
        results_text.insert(tk.END, analysis)

        # Highlight key sections
        highlight_terms = {
            "PROCEED TO INTERVIEW": "green",
            "DECLINE": "red",
            "ADDITIONAL SCREENING": "orange",
            "Key Insights": "blue",
            "Resume vs. Conversation": "blue",
            "Job Fit Analysis": "blue",
            "FINAL RECOMMENDATION": "purple",
            "Confidence Level": "blue",
            "Justification": "blue",
            "High": "green",
            "Medium": "orange",
            "Low": "red"
        }

        for term, color in highlight_terms.items():
            self.highlight_text(results_text, term, color)

        # buttons
        button_frame = tk.Frame(results_window, pady=10)
        button_frame.pack()

        tk.Button(button_frame, text="Save Analysis",
                  command=lambda: self.save_analysis(analysis)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Close",
                  command=results_window.destroy).pack(side=tk.LEFT, padx=10)

    def highlight_text(self, text_widget, phrase, color):
        """Highlight specific text in a text widget"""
        start_pos = "1.0"
        while True:
            start_pos = text_widget.search(phrase, start_pos, stopindex=tk.END)
            if not start_pos:
                break

            end_pos = f"{start_pos}+{len(phrase)}c"
            text_widget.tag_add(phrase, start_pos, end_pos)
            text_widget.tag_config(phrase, foreground=color, font=("Arial", 11, "bold"))
            start_pos = end_pos

    def save_analysis(self, analysis):
        """Save analysis to a text file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis",
            defaultextension=".txt",
            filetypes=[("Text Files", ".txt"), ("All Files", ".*")]
        )

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                metadata = f"""
CANDIDATE EVALUATION REPORT
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Messages analyzed: {len(conversation_messages)}
Initial resume match score: {calculate_match_score(job_description, candidate_resume):.1f}/10

--- ANALYSIS RESULTS ---

"""
                f.write(metadata + analysis)

            messagebox.showinfo("Success", "Analysis saved successfully!")


    def clear_chat(self):
        """Clear the chat history"""
        global conversation_messages

        if messagebox.askyesno("Confirm", "This will clear the entire conversation. Continue?"):
            conversation_messages = []
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.current_sender.set("Recruiter")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()