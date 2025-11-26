

# HTMX + FastAPI + Gemini Streaming Chatbot

A lightweight, high-performance LLM chatbot implementation using **HTMX** for the frontend and **FastAPI** for the backend. This project demonstrates real-time text streaming via **Server-Sent Events (SSE)** and includes a modern, responsive UI built with **Tailwind CSS**.

## Features

*   **Zero-Build Frontend:** No Webpack, no node_modules, no hydration. Uses Tailwind CSS via CDN.
*   **Real-Time Streaming:** Uses SSE to push LLM tokens to the browser instantly.
*   **Modern UI:** Clean, responsive interface with distinct user/bot message styling and animations.
*   **System Instructions:** Configure the AI's persona (e.g., "You are a pirate") via a built-in settings panel. Persists across sessions.
*   **Persistent Chat History:** Uses SQLite (`chat.db`) to save conversations, allowing history to survive server restarts.
*   **Multimodal Support:**
    *   **Image Input:** Upload images to chat about them (uses Gemini Vision).
    *   **Image Output:** The bot can generate images using Pollinations.ai (just ask it to "generate an image").
*   **Web Search:** Integrated with **Tavily API** for real-time information retrieval.
*   **Python Code Execution:** The bot can write and execute Python code for calculations and complex logic.
*   **Date/Time Awareness:** The bot knows the current date and time.
*   **Robust Error Handling:** Gracefully handles API quota limits and empty responses with automatic retries and user notifications.

## Prerequisites

*   Python 3.9+
*   A Google Cloud Project with the [Gemini API enabled](https://aistudio.google.com/).
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
    pip install fastapi uvicorn jinja2 python-dotenv google-generativeai markdown pillow python-multipart tavily-python
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
│   └── uploads/            # Directory for uploaded images
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
*   **Upload Images:** Click the paperclip icon to upload an image.
*   **Generate Images:** Ask "Generate an image of a cat".
*   **Search Web:** Ask "What is the price of Bitcoin?".
*   **Run Code:** Ask "Calculate the 100th Fibonacci number".

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
*   **Image Upload Fails:** Ensure `static/uploads/` directory exists (created automatically).