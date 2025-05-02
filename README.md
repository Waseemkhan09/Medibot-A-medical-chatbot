# 🤖 MediBot - Medical Chatbot

MediBot is an AI-powered medical chatbot integrated within the **MedSense** project. It assists users by answering health-related queries using machine learning models and a medical knowledge base.

---

## 💡 Features

- 🧠 AI-based response generation using **HuggingFace** models and **LangChain**
- 🗂️ Retrieves context from a **FAISS vector store**
- 💬 Streamlit-based UI for chat interaction
- 🔍 Personalized recommendations based on context
- ⚙️ Customizable prompt engineering for answer accuracy

---

## 🚀 How It Works

1. **User types a medical query** into the chatbot input.
2. The query is processed and relevant context is fetched from the vector store.
3. HuggingFace model (Mistral-7B or Blenderbot) generates a human-like response.
4. The response is displayed back to the user in a friendly chat interface.

## 📦 Run Locally

Follow these steps to run the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/WaseemKhan09/Medibot-A-medical-chatbot.git
cd Medibot-A-medical-chatbot
```

### 2. Create & Activate Virtual Environment
```bash
# On Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory with the following:
```env
HF_TOKEN=your_huggingface_api_key_here
```

### 5. Run the Chatbot (Streamlit Interface)
In a new terminal:
```bash
streamlit run medibot.py
---

## Notes
1.Requires an active internet connection to access HuggingFace models.
2.Designed for educational/demo purposes. Do not use for real medical diagnosis.
