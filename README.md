

# HTMX + FastAPI + Gemini Streaming Chatbot

A lightweight, high-performance LLM chatbot implementation using **HTMX** for the frontend and **FastAPI** for the backend. This project demonstrates real-time text streaming via **Server-Sent Events (SSE)** without the complexity of client-side frameworks like React or Vue.

## Features

  * **Zero-Build Frontend:** No Webpack, no node\_modules, no hydration. Pure HTML/CSS.
  * **Real-Time Streaming:** Uses SSE to push LLM tokens to the browser instantly.
  * **Session Management:** Cookie-based sessions allow multiple concurrent users with isolated chat histories.
  * **Rich Text Support:** Server-side Markdown rendering with client-side syntax highlighting (Highlight.js).
  * **Gemini 2.5 Flash:** Optimized for speed and cost-efficiency.

## Prerequisites

  * Python 3.9+
  * A Google Cloud Project with the [Gemini API enabled](https://aistudio.google.com/).

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
    pip install fastapi uvicorn jinja2 python-dotenv google-generativeai markdown
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory:

    ```ini
    GEMINI_API_KEY=your_actual_api_key_here
    ```

## Project Structure

```text
.
├── main.py              # FastAPI server, session logic, and SSE generator
├── .env                 # API Key configuration
├── requirements.txt     # Python dependencies
└── templates/
    └── index.html       # Client UI (HTMX + Tailwind/CSS + Highlight.js)
```

## Running the Application

Start the development server with hot-reloading:

```bash
uvicorn main:app --reload
```

  * **Access the UI:** Open `http://127.0.0.1:8000` in your browser.
  * **Test Multi-user:** Open a second tab in Incognito mode to verify separate session histories.

## Architecture Overview

1.  **Initialization:**

      * Client loads `index.html`.
      * Server assigns a `session_id` cookie if missing.
      * Server initializes an empty `deque` for conversation history linked to that session.

2.  **User Submission:**

      * HTMX sends a `POST /chat`.
      * Server returns HTML immediately containing the User Message and a **Bot Placeholder** (`<div sse-connect="...">`).

3.  **Streaming (SSE):**

      * Browser connects to the `/stream` endpoint defined in the placeholder.
      * Server sends formatted SSE events (`data: ...`).
      * **Intermediate Swaps:** HTMX swaps the content inside the placeholder with incoming HTML chunks.
      * **Formatting:** Server converts Markdown to HTML on the fly.

4.  **Completion (OOB Swap):**

      * Upon stream completion, the server sends a final `hx-swap-oob="outerHTML"` event.
      * This replaces the "live" placeholder with a static `<div>`, effectively closing the SSE connection and rendering the final syntax-highlighted HTML.

## Troubleshooting

  * **Connection Loop:** If the chat constantly flickers or re-requests, ensure the server is sending the final `hx-swap-oob` event to replace the connecting element.
  * **404 on Model:** Ensure you are using a valid model name in `main.py` (e.g., `gemini-2.5-flash` or `gemini-1.5-flash`).
  * **No Styles:** Ensure the CSS in `index.html` is present and `highlight.js` CDN links are reachable.