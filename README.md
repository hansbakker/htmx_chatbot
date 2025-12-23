

# HTMX + FastAPI + Gemini/OpenAI Streaming Chatbot

A lightweight, high-performance LLM chatbot implementation using **HTMX** for the frontend and **FastAPI** for the backend. This project demonstrates real-time text streaming via **Server-Sent Events (SSE)** and includes a modern, responsive UI built with **Tailwind CSS**.

## Features

### Core Capabilities
*   **Multi-User Support (Google SSO):** Integrated Google OAuth 2.0 login with secure data segregation and persistent sessions.
*   **Real-Time Streaming:** Uses SSE to push LLM tokens to the browser instantly.
*   **Modern UI:** A clean, responsive "glassmorphism" interface with dark mode support and smooth animations.
*   **Chat Management:** Full lifecycle management including Rename, Archive, and Delete conversations.
*   **Persistent History:** All conversations and settings are stored in a local SQLite database (`chat.db`).

### Advanced LLM Interaction
*   **Provider Agnostic:** Supports both **Google Gemini** and **OpenAI** (configurable via environment variables).
*   **Model Picker:** Switch between different models (e.g., Gemini 1.5 Pro/Flash, GPT-4o) directly from the UI.
*   **Persona Configuration:** Custom "System Instructions" that persist across sessions.
*   **Auto-Continue:** Automatically handles multi-step reasoning or long tasks by detecting "thinking" phrases.

### Multimodal & File Support
*   **Generic File Upload:** Analyze PDFs, CSVs, images, and text files.
*   **Vision Support:** Direct image/video analysis for supported models.
*   **File Context Management:** Intelligent history management when analyzing files to optimize token usage and compatibility.

### Intelligent Tooling & Execution
*   **Secure Code Sandbox (E2B):** Run Python calculations and data visualizations (Matplotlib/Plotly) in a secure, isolated cloud environment.
*   **Web Intel:** Integrated web search (Tavily) and deep website crawling.
*   **Developer Tools:** Tools to read, write, and modify the chatbot's own source code or install new Python packages at runtime.
*   **Utility Tools:** PDF generation, weather data (XWeather), timezone-aware datetime, and location services (Geopy).

## Prerequisites

*   Python 3.12+
*   **API Keys:**
    *   [Gemini API Key](https://aistudio.google.com/) and/or [OpenAI API Key](https://platform.openai.com/).
    *   [Tavily API Key](https://tavily.com/) for web search/crawling.
    *   [E2B API Key](https://e2b.dev/) for secure code execution (recommended).
*   **Google Cloud Project:** Required for Google SSO (OAuth 2.0 Client ID/Secret).
*   **System Dependencies (macOS):** `brew install cairo pango` (for PDF generation).

## Installation

1.  **Clone the project:**
    ```bash
    git clone <repository-url>
    cd htmx-chatbot
    ```

2.  **Setup Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install fastapi uvicorn jinja2 python-dotenv google-generativeai openai markdown pillow python-multipart tavily-python geopy matplotlib numpy pandas requests plotly kaleido authlib itsdangerous httpx xhtml2pdf e2b_code_interpreter
    ```
    or use the requirements.txt file
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    Create a `.env` file based on the following template:
    ```ini
    # Providers
    GEMINI_API_KEY=...
    OPENAI_API_KEY=...
    LLM_PROVIDER=gemini # or 'openai'
    
    # Tools
    TAVILY_API_KEY=...
    E2B_API_KEY=...
    WOLFRAM_ALPHA_APP_ID=...
    XWEATHER_MCP_ID=...
    XWEATHER_MCP_SECRET=...
    
    # Auth & Security
    GOOGLE_CLIENT_ID=...
    GOOGLE_CLIENT_SECRET=...
    SECRET_KEY=... # Random string for sessions
    # OAUTHLIB_INSECURE_TRANSPORT=1 # For local HTTP dev
    ```

## Project Structure

```text
.
├── main.py                 # Core FastAPI application & tool definitions
├── providers/              # LLM Provider abstraction layer (Gemini, OpenAI)
├── chat.db                 # SQLite database (History, Settings, Users)
├── static/
│   ├── uploads/            # User-uploaded files
│   ├── generated/          # Generated charts, PDFs, and images
│   └── generated_code/     # Files written by the 'write_source_code' tool
├── templates/
│   └── index.html          # Frontend UI (HTMX + Tailwind)
└── system_instruction.txt  # Persisted AI persona
```

## Available Tools

The AI assistant can autonomously use the following tools:

1.  **`search_web(query, count)`**: Web search via Tavily.
2.  **`crawl_website(url, depth, limit)`**: deep website content extraction.
3.  **`execute_calculation(code)`**: Python math/logic (Numpy/Pandas). Supports E2B Sandbox.
4.  **`generate_chart(code)`**: Matplotlib visualizations (PNG). Supports E2B Sandbox.
5.  **`generate_plotly_chart(code)`**: Interactive Plotly visualizations (HTML + PNG).
6.  **`generate_image(prompt)`**: Artistic image generation via Pollinations.ai.
7.  **`convert_md_to_pdf(md)`**: Create downloadable PDF documents.
8.  **`read_uploaded_file(name)`**: Access text content of user uploads.
9.  **`write_source_code(path, code)`**: Modify or create local source files.
10. **`read_source_code(path)`**: Read local source files.
11. **`import_package(pkg)`**: Dynamically install/import Python packages.
12. **`wolfram_alpha_query(q)`**: Scientific & factual data engine.
13. **`get_weather(loc)`**: Current conditions and multi-day forecasts.
14. **`get_current_datetime(tz)`**: Timezone-aware date/time detection.
15. **`chat_management`**: Tools to rename, archive, or delete conversations.

## Advanced Configuration

### Code Execution Mode
In **Settings**, you can toggle between **Local** and **Sandbox** execution. 
*   **Local:** Uses the server's Python environment (fast, but less secure).
*   **Sandbox (E2B):** Runs code in an isolated cloud environment with a configurable timeout (default 300s). This is recommended for safety and dependency isolation.

### File Context Behavior
When a file is uploaded, the chatbot may temporarily clear conversation history to ensure the model focuses on the file content and stays within token limits. Use the **"Clear File Context"** button to restore standard history and tools after your analysis is complete.

## Troubleshooting

*   **OAuth Redirects:** Ensure your Authorized Redirect URI in Google Console matches your local URL (e.g., `http://localhost:8000/auth`). If you are using a custom domain, ensure the redirect URI is set to `https://<your-domain>/auth`. Or use 192-168-X-X.sslips.io/auth (replace X with your local IP address). in that url of  the chatbot the address will be http://192-168-X-X.sslips.io/ 
*   **Timezone Detection:** Localhost (`127.0.0.1`) geolocates to UTC. Deploy to a public IP for accurate automatic detection.
*   **PDF Errors:** On Windows/Linux, ensure appropriate system-level headers for `cairo` are installed.