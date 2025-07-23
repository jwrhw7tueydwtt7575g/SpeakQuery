# 🎙️ SpeakQuery

SpeakQuery is an AI-powered voice-to-query document search and summarization system. It leverages voice input to perform semantic search across uploaded documents using a RAG (Retrieval-Augmented Generation) approach with LLMs from Groq and local embedding models.

---

## 🚀 Features

- 🎤 **Voice Input**: Speak your query and convert it to text using speech recognition.
- 📄 **Multi-format Support**: Upload PDF, PPTX, Excel (XLSX/CSV), and TXT documents.
- 🧠 **Semantic Search**: Hybrid search using dense embeddings + sparse TF-IDF scoring.
- 📝 **Smart Summaries**: Extract and summarize content using Groq LLMs (LLaMA3).
- 🗂️ **Multi-file Support**: Upload and search across multiple documents at once.
- 🔐 **Secure API Keys**: All secrets handled via `.env`.

---

## 🛠️ Tech Stack

- Python (Flask)
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Groq API (`llama3-8b-8192`)
- SpeechRecognition (`Google Web Speech API`)
- FFmpeg (for audio conversion)
- TfidfVectorizer + Cosine Similarity
- HTML/CSS (Flask templating)

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/SpeakQuery.git
cd SpeakQuery
2. Set up a virtual environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate   # On macOS/Linux
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add .env file
Create a .env file in the root directory:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
▶️ Usage
1. Run the Flask server
bash
Copy
Edit
python app.py
2. Open in browser
Visit: http://localhost:5000

3. Upload documents and speak your query!
📁 File Structure
bash
Copy
Edit
.
├── app.py               # Main Flask app
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Frontend interface
├── uploads/             # Uploaded files
├── .env                 # Secret keys (excluded via .gitignore)
└── README.md            # You're here!
🔒 Security Notes
Do not hardcode API keys. Use the .env file instead.

Your .env should be listed in .gitignore to prevent accidental commits.

GitHub Push Protection may block pushes with exposed secrets.

📄 License
MIT License. Use freely, but please credit this repo if you build on top of it.

