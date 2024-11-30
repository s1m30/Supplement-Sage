# 📝 Supplement Sage

**Your AI-powered assistant for mastering college supplemental essays.**

Supplement Sage is a web application designed to help students navigate the complexities of college applications. With AI-driven tools and personalized assistance, users can explore, organize, and refine their essays while integrating data from multiple sources.

---
![WhatsApp Image 2024-11-26 at 13 18 12](https://github.com/user-attachments/assets/e85d58af-4881-4322-b656-b6fa7e618366)

## 🚀 Features

- **Streamlined Sources**: Upload files, add web links, or YouTube URLs for analysis and extraction.
- **AI-Powered Assistance**: Generate and refine supplemental essay responses tailored to specific colleges.
- **Supabase Authentication**: Each user has their own section in the database for secure and private access.
---

## 🛠️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Backend**: [LangChain](https://langchain.com/), [Supabase](https://supabase.io/)
- **AI Models**: HuggingFace Transformers for embeddings and natural language processing.
- **Database**: PostgreSQL with pgvector for embeddings.
---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Supabase account
- API keys for LangChain and HuggingFace

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/supplementsage.git
   cd supplementsage
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Create a `.env` file:
     ```plaintext
     SUPABASE_URL=<your_supabase_url>
     SUPABASE_KEY=<your_supabase_api_key>
     LANGCHAIN_API_KEY=<your_langchain_api_key>
     HUGGINGFACE_API_TOKEN=<your_huggingface_api_token>
     ```
   - Replace placeholder values with your credentials.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🧩 Usage

1. **Login**: Authenticate using your email/password via Supabase.
2. **Upload Sources**: Upload files or enter URLs for analysis.
3. **Add Questions**: Select colleges and add supplemental essay questions to the vector store.

---

## 🌟 Acknowledgments

- [Streamlit](https://streamlit.io/) for its amazing UI capabilities.
- [LangChain](https://langchain.com/) for powering the AI workflows.
- [Supabase](https://supabase.io/) for robust database and authentication.
- The college admissions community for inspiration.

--- 

Let me know if you want modifications or have specific details you'd like to include!
