

# HTMX + FastAPI + Gemini Streaming Chatbot

A lightweight, high-performance LLM chatbot implementation using **HTMX** for the frontend and **FastAPI** for the backend. This project demonstrates real-time text streaming via **Server-Sent Events (SSE)** and includes a modern, responsive UI built with **Tailwind CSS**.

## Features

*   **Zero-Build Frontend:** No Webpack, no node_modules, no hydration. Uses Tailwind CSS via CDN.
*   **Real-Time Streaming:** Uses SSE to push LLM tokens to the browser instantly.
*   **Modern UI:** Clean, responsive interface with distinct user/bot message styling and animations.
*   **System Instructions:** Configure the AI's persona (e.g., "You are a pirate") via a built-in settings panel. Persists across sessions.
*   **Model Selection:** Choose from multiple Gemini models (Flash, Pro, Flash Thinking) via the settings panel.
*   **Persistent Chat History:** Uses SQLite (`chat.db`) to save conversations, allowing history to survive server restarts.
*   **Multimodal Support:**
    *   **File Input:** Upload any file type (PDFs, CSVs, text files, images, etc.) to analyze and ask questions about them.
    *   **Image Output:** The bot can generate images using Pollinations.ai (just ask it to "generate an image").
    *   **Chart/Graph Generation:** The bot can create data visualizations (pie charts, bar charts, line graphs, etc.) using matplotlib. Generated charts are automatically saved and displayed.
*   **File Context Management:** 
    *   When analyzing files, conversation history is temporarily cleared to ensure API compatibility.
    *   Use the "Clear File Context" button to restore tools and conversation history after file analysis.
*   **Web Search:** Integrated with **Tavily API** for real-time information retrieval.
*   **Python Code Execution:** The bot can write and execute Python code for calculations and complex logic. Supports all standard Python libraries including matplotlib, numpy, etc.
*   **Date/Time Awareness:** The bot knows the current date and time.
*   **Location Services:** The bot can find coordinates for locations using geopy.
*   **Robust Error Handling:** Gracefully handles API quota limits, token limits, and empty responses with automatic retries and user notifications.

## Prerequisites

*   Python 3.9+
*   A [Gemini API Key](https://aistudio.google.com/).
*   A [Tavily API Key](https://tavily.com/) for web search.

## Installation

1.  **Clone or create the project directory:**

    ```bash
    mkdir htmx-chatbot
    cd htmx-chatbot
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file (or run directly):

    ```bash
    pip install fastapi uvicorn jinja2 python-dotenv google-generativeai markdown pillow python-multipart tavily-python geopy matplotlib numpy
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory:

    ```ini
    GEMINI_API_KEY=your_gemini_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    ```

## Project Structure

```text
.
├── main.py                 # FastAPI server, session logic, tools, and SSE generator
├── .env                    # API Key configuration
├── requirements.txt        # Python dependencies
├── system_instruction.txt  # Persisted system prompt (persona)
├── chat.db                 # SQLite database for chat history
├── .gitignore              # Git ignore file
├── static/
│   ├── uploads/            # Directory for uploaded images
│   └── generated/          # Directory for generated charts/plots
└── templates/
    └── index.html          # Client UI (HTMX + Tailwind CSS + Highlight.js)
```

## Running the Application

Start the development server with hot-reloading:

```bash
uvicorn main:app --reload
```

*   **Access the UI:** Open `http://127.0.0.1:8000` in your browser.
*   **Configure Persona:** Click the "Settings" button in the top right to change the AI's system instruction.
*   **Select Model:** In the settings panel, choose your preferred Gemini model (Flash, Pro, or Flash Thinking).
*   **Upload Files:** Click the paperclip icon to upload any file (images, PDFs, CSVs, text files, etc.).
*   **Analyze Files:** After uploading, ask questions about the file content.
*   **Clear File Context:** Click the "📎 Clear File Context" button to remove file references and restore tools/history.
*   **Generate Images:** Ask "Generate an image of a cat" (artistic).
*   **Generate Charts:** Ask "Generate a pie chart showing browser usage" (data visualization).
*   **Search Web:** Ask "What is the price of Bitcoin?".
*   **Run Code:** Ask "Calculate the 100th Fibonacci number".
*   **Get Time/Date:** Ask "What time is it?" or "What's today's date?".

## Architecture Overview

1.  **Initialization:**
    *   Client loads `index.html`.
    *   Server assigns a `session_id` cookie if missing.
    *   Server loads chat history from `chat.db`.

2.  **User Submission:**
    *   HTMX sends a `POST /chat` (multipart/form-data for images).
    *   Server saves user message to DB.
    *   Server returns HTML immediately containing the User Message and a **Bot Placeholder** (`<div sse-connect="...">`).

3.  **Streaming (SSE) & Tools:**
    *   Browser connects to the `/stream` endpoint.
    *   Server initializes Gemini with tools (`search_web`, `execute_python`, `get_current_datetime`).
    *   **Function Calling:** If the model calls a tool, the server executes it (e.g., runs Python code), feeds the result back to the model, and streams the final answer.
    *   **Retries:** Automatic retries for empty responses or transient errors.
    *   **Formatting:** Server converts Markdown to HTML on the fly.

4.  **Completion (OOB Swap):**
    *   Upon stream completion, the server saves the bot's response to DB.
    *   Server sends a final `hx-swap-oob="outerHTML"` event to finalize the UI.

## Troubleshooting

*   **Quota Exceeded:** If you see a red "Quota Exceeded" error, wait a minute and try again. The app handles this gracefully.
*   **No Search Results:** Ensure `TAVILY_API_KEY` is set in `.env`.
*   **File Upload Fails:** Ensure `static/uploads/` directory exists (created automatically).
*   **Tools Not Working After File Upload:** When a file is in context, tools are automatically disabled. Click "📎 Clear File Context" to re-enable them.
*   **Lost Conversation History:** File analysis temporarily clears history for API compatibility. Use "Clear File Context" to restore history when done analyzing files.