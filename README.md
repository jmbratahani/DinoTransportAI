# DinoTransportAI
LLM Driven ChatBot to assist with Dinosaur Transport Safety Operations

## Project Overview
This project, **DinoTransport Copilot**, is developed as part of a project for the graduate level course**DSO 599 at USC Marshall School of Business** by Xiaoyi Guan, Jianan Ding, Jessica Bratahani. The application is a **Streamlit-based chatbot** that integrates **LangChain** and various AI tools to assist users with dinosaur transport safety-related queries.

## Features
- **PDF Document Search**: Utilizes **LangChain**'s `PyPDFDirectoryLoader` to load and search dinosaur-related PDFs.
- **Vector Database for Information Retrieval**: Implements **FAISS** and `SentenceTransformerEmbeddings` to enable fast and efficient searching within documents.
- **AI Chatbot with OpenAI**: Uses `ChatOpenAI` (GPT-based) to process user queries and provide intelligent responses.
- **Weather Data Retrieval**: Integrates the **OpenWeatherMap API** to fetch real-time weather information for transportation considerations.
- **Email Notification via AWS SES**: Implements an **AWS Lambda function** to send email notifications related to dinosaur transport.
- **SQL Database Integration**: Uses **SQLite** to store and retrieve structured dinosaur transport data.
- **Conversational Memory**: Implements **session-based chat history** for a more interactive and context-aware experience.

## Tech Stack
- **Python**
- **Streamlit** (for web app UI)
- **LangChain** (for AI-driven chatbot functionalities)
- **FAISS** (for vector database)
- **OpenAI GPT API** (for AI responses)
- **OpenWeatherMap API** (for weather retrieval)
- **AWS Lambda & SES** (for email notifications)
- **SQLite** (for structured data management)

## How It Works
1. **Initialization** (`initial()`)
   - Loads PDF documents and converts them into searchable text chunks.
   - Creates a vector store using FAISS for efficient retrieval.
   - Sets up OpenAI-based AI model for chatbot responses.
   - Integrates weather API and email notification functionalities.
   - Configures SQLite database for structured data access.

2. **Dinosaur Transport Assistance** (`main()`)
   - Users enter queries about dinosaur transport safety.
   - The chatbot processes the query and retrieves relevant information.
   - Responses are displayed in an interactive UI.
   - Weather and email functionalities are available as additional tools.

## Setup Instructions
1. Install required dependencies:
   ```bash
   pip install streamlit langchain langchain-community faiss-cpu openai sqlite3
   ```
2. Set environment variables (replace with your actual keys):
   ```bash
   export OPENWEATHERMAP_API_KEY="your_api_key_here"
   ```
3. Run the application:
   ```bash
   streamlit run dino_copilot.py
   ```

## Future Improvements
- Enhance chatbot with more advanced **memory handling**.
- Extend **document retrieval** with additional data sources.
- Integrate **real-time tracking** of dinosaur transport routes.

---
Developed by **Xiaoyi Guan, Jianan Ding, Jessica Bratahani** for **DSO 599 Project 2 - USC Spring 2024**.


