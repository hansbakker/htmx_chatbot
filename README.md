

# HTMX + FastAPI + Gemini Streaming Chatbot

A lightweight, high-performance LLM chatbot implementation using **HTMX** for the frontend and **FastAPI** for the backend. This project demonstrates real-time text streaming via **Server-Sent Events (SSE)** and includes a modern, responsive UI built with **Tailwind CSS**.

## Features

*   **Zero-Build Frontend:** No Webpack, no node_modules, no hydration. Uses Tailwind CSS via CDN.
*   **Real-Time Streaming:** Uses SSE to push LLM tokens to the browser instantly.
*   **Modern UI:** Clean, responsive interface with distinct user/bot message styling and animations.
*   **Dark Mode:** Toggle between light and dark themes with persistent preference storage.
*   **Chat History & Sidebar:**
    *   Sidebar navigation showing all previous conversations.
    *   **Context Menu:** Rename, Archive, or Delete chats directly from the sidebar.
    *   Auto-saved chat sessions with smart naming based on first message.
    *   Easy switching between conversations.
    *   "New Chat" button to start fresh conversations.
    *   **Archived Chats:** View and manage archived conversations via the Settings panel.
*   **Smart Scroll Behavior:** Intelligently positions user queries at the top while showing as much of the response as possible.
*   **System Instructions:** Configure the AI's persona (e.g., "You are a pirate") via a built-in settings panel. Persists across sessions.
*   **Model Selection:** Choose from multiple Gemini models (Flash, Pro, Flash Thinking) via the settings panel.
*   **Persistent Chat History:** Uses SQLite (`chat.db`) to save conversations, allowing history to survive server restarts.
*   **Multimodal Support:**
    *   **File Input:** Upload any file type (PDFs, CSVs, text files, images, etc.) to analyze and ask questions about them.
    *   **Image Generation:** Generate artistic images using Pollinations.ai via the `generate_image` tool.
    *   **Chart/Graph Generation:** Create data visualizations (bar charts, line graphs, scatter plots, etc.) using matplotlib via the `generate_chart` tool.
*   **File Context Management:** 
    *   When analyzing files, conversation history is temporarily cleared to ensure API compatibility.
    *   Use the "Clear File Context" button to restore tools and conversation history after file analysis.
*   **Intelligent Tools:**
    *   **Web Search:** Integrated with **Tavily API** for real-time information retrieval.
    *   **Python Execution:** Separate tools for calculations (`execute_calculation`) and visualizations (`generate_chart`).
    *   **Weather Services:** Real-time weather data via **XWeather MCP** (current conditions, forecasts, precipitation timing).
    *   **Timezone Awareness:** Automatically detects user timezone from IP address, with manual override support.
    *   **Date/Time:** Get current time in user's timezone or any specified timezone.
    *   **Location Services:** Find coordinates for locations using geopy.
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
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file (or run directly):

    ```bash
    pip install fastapi uvicorn jinja2 python-dotenv google-generativeai markdown pillow python-multipart tavily-python geopy matplotlib numpy pandas requests
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory:

    ```ini
    GEMINI_API_KEY=your_gemini_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    XWEATHER_MCP_ID=your_xweather_client_id_here
    XWEATHER_MCP_SECRET=your_xweather_client_secret_here
    ```
    
    **Note:** XWeather credentials are optional. If not provided, weather-related tools will not be available.

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
│   ├── uploads/            # Directory for uploaded files
│   └── generated/          # Directory for generated charts/images
└── templates/
    └── index.html          # Client UI (HTMX + Tailwind CSS + Highlight.js)
```

## Running the Application

Start the development server with hot-reloading:
```bash
uvicorn main:app --reload
```

*   **Access the UI:** Open `http://127.0.0.1:8000` in your browser.
*   **Dark Mode:** Click the moon/sun icon in the header to toggle between light and dark themes.
*   **Chat History:** View all your previous conversations in the left sidebar. Click any chat to load it.
*   **Manage Chats:** Hover over a chat in the sidebar and click the three-dot menu (⋮) to **Rename**, **Archive**, or **Delete** it.
*   **Archived Chats:** Go to Settings > Archived Chats to view, unarchive, or permanently delete hidden conversations.
*   **New Chat:** Click the "New Chat" button in the sidebar (or header on mobile) to start a fresh conversation.
*   **Configure Persona:** Click the "Settings" button in the top right to change the AI's system instruction.
*   **Select Model:** In the settings panel, choose your preferred Gemini model (Flash, Pro, or Flash Thinking).
*   **Upload Files:** Click the paperclip icon to upload any file (images, PDFs, CSVs, text files, etc.).
*   **Analyze Files:** After uploading, ask questions about the file content.
*   **Clear File Context:** Click the "📎 Clear File Context" button to remove file references and restore tools/history.
*   **Generate Images:** Ask "Draw a cat wearing a wizard hat" (artistic images via Pollinations.ai).
*   **Generate Charts:** Ask "Create a bar chart showing Q1=100, Q2=150, Q3=120, Q4=180" (data visualization via matplotlib).
*   **Search Web:** Ask "What is the price of Bitcoin?".
*   **Run Calculations:** Ask "Calculate the 100th Fibonacci number".
*   **Get Weather:** Ask "What's the weather in London?" or "What's the forecast for next week in Paris?".
*   **Get Time/Date:** Ask "What time is it?" (automatically detects your timezone) or "What time is it in Tokyo?".
*   **Get Timezone:** Ask "What is my timezone?".

## Architecture Overview

1.  **Initialization:**
    *   Client loads `index.html`.
    *   Server assigns a `session_id` cookie if missing.
    *   Server loads chat history from `chat.db`.

2.  **User Submission:**
    *   HTMX sends a `POST /chat` (multipart/form-data for files).
    *   Server saves user message to DB.
    *   Server returns HTML immediately containing the User Message and a **Bot Placeholder** (`<div sse-connect="...">`)`.

3.  **Streaming (SSE) & Tools:**
    *   Browser connects to the `/stream` endpoint.
    *   Server initializes Gemini with tools (`search_web`, `execute_calculation`, `generate_chart`, `generate_image`, `get_current_datetime`, `get_user_timezone`, `get_coordinates`).
    *   **Function Calling:** If the model calls a tool, the server executes it (e.g., generates a chart, searches the web), feeds the result back to the model, and streams the final answer.
    *   **Timezone Detection:** Client IP is captured and used for automatic timezone detection in datetime queries.
    *   **Retries:** Automatic retries for empty responses or transient errors.
    *   **Formatting:** Server converts Markdown to HTML on the fly.

4.  **Completion (OOB Swap):**
    *   Upon stream completion, the server saves the bot's response to DB.
    *   Server sends a final `hx-swap-oob="outerHTML"` event to finalize the UI.

## Available Tools

The chatbot has access to the following tools:

1.  **`search_web(query)`**: Searches the internet using Tavily API.
2.  **`execute_calculation(code)`**: Executes Python code (numpy/pandas) for math/logic.
3.  **`generate_chart(code)`**: Creates visualizations using matplotlib.
4.  **`generate_image(description)`**: Generates artistic images via Pollinations.ai.
5.  **`get_current_datetime(timezone)`**: Returns current date/time (timezone-aware).
6.  **`get_user_timezone()`**: Returns the user's detected timezone.
7.  **`get_coordinates(location)`**: Returns latitude/longitude for a location.
8.  **`get_weather(location)`**: Returns current weather for a location via XWeather MCP.
9.  **`get_forecast_weather(location)`**: Returns multi-day weather forecast via XWeather MCP.
10. **`get_precipitation_timing(location)`**: Returns rain/snow start/stop timing via XWeather MCP.

## Troubleshooting

*   **Timezone Detection on Localhost:** When running locally (`127.0.0.1`), the IP detection will see a local IP (e.g., `127.0.0.1`) which cannot be geolocated. In this case, the system defaults to `UTC`. To test timezone detection, you can manually set the `X-Forwarded-For` header or deploy to a public server.
*   **XWeather API Errors:** If you see "invalid location" errors, try specifying the location more precisely (e.g., "City, Country" or Zip Code). Ensure `XWEATHER_MCP_ID` and `XWEATHER_MCP_SECRET` are set in `.env`.
*   **File Upload Fails:** Ensure `static/uploads/` directory exists (created automatically).
*   **Tools Not Working After File Upload:** When a file is in context, tools are automatically disabled. Click "📎 Clear File Context" to re-enable them.
*   **Lost Conversation History:** File analysis temporarily clears history for API compatibility. Use "Clear File Context" to restore history when done analyzing files.
*   **Timezone Not Detected:** For localhost (127.0.0.1), timezone detection defaults to UTC. Deploy to a server with a public IP for automatic detection.