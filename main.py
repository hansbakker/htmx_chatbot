import os
import asyncio
import uuid
import sqlite3
import markdown
import io
import sys
import traceback
import contextlib
import datetime
import shutil
import contextvars
import urllib.request
import json
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for older Python versions if needed, though 3.9+ is expected
    ZoneInfo = None

from collections import deque

# to allow importing packages during runtime
import subprocess
import importlib
   
from typing import Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, Response, Cookie, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tavily import TavilyClient
import urllib.parse
import json
from geopy.geocoders import Nominatim
from google.protobuf import struct_pb2

# Configure Matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

# Configure Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Context variable to store client IP
client_ip_ctx = contextvars.ContextVar("client_ip", default=None)
session_id_ctx = contextvars.ContextVar("session_id", default=None)

# Retry phrases that trigger auto-continue (case-insensitive check)
RETRY_PHRASES = [
    "please wait",
    "please give me a moment",
    "let me try",
    "i'll try",
    "i will try",
    "bear with me",
    "i'll proceed",
    "i will proceed",
    "i'll get",
    "i will get",
    "i'll wait",
    "i will wait",
    "i'll stop",
    "i will stop",
    "working on it",
    "one moment",
    "give me a moment",
    "let me fix",
    "i'll fix",
    "i will fix",
    "i'll adjust",
    "i will adjust",
    "i'll retry",
    "i will retry",
    "i will",
    "stand by",
    "might take a few moments",
    "here we go"
]

# --- 1. CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Remove global model instantiation to allow dynamic system instructions
# Configure Gemini
# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configure Matplotlib for headless environment
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    plt = None
    np = None
    pd = None
    print("Warning: matplotlib, numpy, or pandas not installed. Chart generation will be unavailable.")

# Configure Tavily
tavily_client = None
if "TAVILY_API_KEY" in os.environ:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    print(f"Tavily Search: Enabled (Key found: {os.environ['TAVILY_API_KEY'][:4]}...)")
else:
    print("Tavily Search: Disabled (TAVILY_API_KEY not found in env)")

SYSTEM_INSTRUCTION_FILE = "system_instruction.txt"

def get_system_instruction() -> str:
    if os.path.exists(SYSTEM_INSTRUCTION_FILE):
        with open(SYSTEM_INSTRUCTION_FILE, "r") as f:
            return f.read().strip()
    instruction = """you are a helpfull AI assistant and you:
- offer specific follow up actions, no general suggestions like "Would you like to know anything else?" 
- use metric system for measurements,
- use Centigrade for temperature,   
- my location is : "Utrecht, The Netherlands".  use that location for any queries that relate to the 'implied' current location of the user (i.e. 'here') when approximate or precise location is needed to answer the question,
"""
    save_system_instruction(instruction)
    return instruction

def save_system_instruction(instruction: str):
    with open(SYSTEM_INSTRUCTION_FILE, "w") as f:
        f.write(instruction)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- 2. DATABASE & SESSION MANAGEMENT ---
DB_NAME = "chat.db"
DB_TIMEOUT = 60.0  # Increased timeout for database operations
GENERATED_DIR = "static/generated"  # Directory for generated charts/images

# Retry decorator for database operations
def retry_on_db_lock(max_retries=5, delay=0.1):
    """Decorator to retry database operations if they fail due to locking."""
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < max_retries - 1:
                        print(f"Database locked in {func.__name__}, retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            return None
        return wrapper
    return decorator

def init_db():
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_archived INTEGER DEFAULT 0
            )
        """)
        
        # Migration: Add is_archived column if it doesn't exist
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN is_archived INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Column likely exists

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                mime_type TEXT,
                FOREIGN KEY(session_id) REFERENCES conversations(id)
            )
        """)
        # Migration for existing table
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN image_path TEXT")
        except sqlite3.OperationalError:
            pass # Column likely exists
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN mime_type TEXT")
        except sqlite3.OperationalError:
            pass # Column likely exists
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN gemini_uri TEXT")
        except sqlite3.OperationalError:
            pass # Column likely exists
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Populate conversations from existing messages if needed
        cursor = conn.execute("SELECT DISTINCT session_id FROM messages WHERE session_id NOT IN (SELECT id FROM conversations)")
        existing_sessions = cursor.fetchall()
        for (session_id,) in existing_sessions:
            # Get first message for title
            cursor = conn.execute("SELECT content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id ASC LIMIT 1", (session_id,))
            first_msg = cursor.fetchone()
            title = "New Chat"
            if first_msg:
                title = first_msg[0][:30] + "..." if len(first_msg[0]) > 30 else first_msg[0]
            
            conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (session_id, title))
        
        print("Database initialized with WAL mode enabled")

# Initialize DB on startup
init_db()

def get_conversations(limit: int = 50, include_archived: bool = False):
    """Get conversations, optionally filtering out archived ones."""
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        conn.row_factory = sqlite3.Row
        if include_archived:
            cursor = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,))
        else:
            cursor = conn.execute("SELECT * FROM conversations WHERE is_archived = 0 ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

def delete_chat_files(session_id: str):
    """Delete all files associated with a chat session."""
    import os
    
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        # Get generated files
        cursor = conn.execute("SELECT file_path FROM generated_files WHERE session_id = ?", (session_id,))
        generated_files = [row[0] for row in cursor.fetchall()]
        
        # Get uploaded files
        cursor = conn.execute("SELECT image_path FROM messages WHERE session_id = ? AND image_path IS NOT NULL", (session_id,))
        uploaded_files = [row[0] for row in cursor.fetchall()]
        
        # Delete physical files
        all_files = generated_files + uploaded_files
        for file_path in all_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Delete database records
        conn.execute("DELETE FROM generated_files WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
        conn.commit()

@retry_on_db_lock()
def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieves history for a specific session ID from DB."""
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        conn.row_factory = sqlite3.Row
        # Select gemini_uri too
        cursor = conn.execute(
            "SELECT role, content, image_path, gemini_uri FROM messages WHERE session_id = ? ORDER BY id ASC", 
            (session_id,)
        )
        rows = cursor.fetchall()

    history = []
    for row in rows:
        role = row["role"]
        content = row["content"]
        parts = []
        
        # Check if content is JSON (for tool calls/responses)
        if content.strip().startswith("{"):
            try:
                data = json.loads(content)
                if "function_call" in data:
                    # Reconstruct FunctionCall part
                    fc_data = data["function_call"]
                    parts.append(
                        genai.protos.Part(
                            function_call=genai.protos.FunctionCall(
                                name=fc_data["name"],
                                args=fc_data["args"]
                            )
                        )
                    )
                elif "function_response" in data:
                    # Reconstruct FunctionResponse part
                    fr_data = data["function_response"]
                    parts.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fr_data["name"],
                                response=fr_data["response"]
                            )
                        )
                    )
                else:
                    # Fallback for other JSON or plain text
                    parts.append(content)
            except json.JSONDecodeError:
                parts.append(content)
        else:
            parts.append(content)

        # Add image if present
        image_path = row["image_path"]
        # Check if gemini_uri exists in keys (for safety during migration)
        gemini_uri = row["gemini_uri"] if "gemini_uri" in row.keys() else None
        mime_type = row["mime_type"] if "mime_type" in row.keys() else "image/jpeg" # Default fallback
        
        if gemini_uri:
            # Use genai.protos.Part with FileData
            parts.append(
                genai.protos.Part(
                    file_data=genai.protos.FileData(
                        mime_type=mime_type,
                        file_uri=gemini_uri
                    )
                )
            )
        elif image_path:
            # Fallback: We have a local path but no URI. 
            pass

        history.append({"role": role, "parts": parts})
    return history

@retry_on_db_lock()
def save_message(session_id: str, role: str, content: str, image_path: Optional[str] = None, mime_type: Optional[str] = None, gemini_uri: Optional[str] = None):
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        # Ensure conversation exists
        cursor = conn.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
        if not cursor.fetchone():
            title = "New Chat"
            if role == "user":
                title = content[:30] + "..." if len(content) > 30 else content
            conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (session_id, title))
        elif role == "user":
             # Update updated_at
             conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
             
             # Update title if it's still generic
             cursor = conn.execute("SELECT title FROM conversations WHERE id = ?", (session_id,))
             current_title = cursor.fetchone()[0]
             if current_title == "New Chat":
                 new_title = content[:30] + "..." if len(content) > 30 else content
                 conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, session_id))

        conn.execute(
            "INSERT INTO messages (session_id, role, content, image_path, mime_type, gemini_uri) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, image_path, mime_type, gemini_uri)
        )

# --- 3. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request, response: Response):
    """
    On first load, check for a cookie. If missing, generate one.
    """
    # Check if user already has a session cookie
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # Load history to render
    chat_history_html = ""
    
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT role, content, image_path FROM messages WHERE session_id = ? ORDER BY id ASC", 
            (session_id,)
        )
        rows = cursor.fetchall()
        
        for row in rows:
            role = row["role"]
            content = row["content"]
            image_path = row["image_path"]

            # Handle internal function responses
            if role == "function":
                try:
                    fr_data = json.loads(content)
                    fn_name = fr_data.get("function_response", {}).get("name", "")
                    fn_response = fr_data.get("function_response", {}).get("response", {}).get("result", "")
                    
                    # Render sources for search_web
                    if fn_name == "search_web":
                        try:
                            # The result itself is a JSON string containing context and sources
                            search_data = json.loads(fn_response)
                            sources = search_data.get("sources", [])
                            
                            if sources:
                                chat_history_html += """
                                <div class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                                    <details class="group">
                                        <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                                            <span class="transition group-open:rotate-90">▶</span>
                                            <span>🔍 Sources used for web search</span>
                                        </summary>
                                        <div class="mt-3 space-y-2">
                                """
                                
                                for idx, source in enumerate(sources, 1):
                                    url = source.get('url', '')
                                    title = source.get('title', url)
                                    chat_history_html += f"""
                                    <div class="flex items-start gap-2 text-sm">
                                        <span class="text-gray-500 dark:text-gray-400 font-mono">{idx}.</span>
                                        <a href="{url}" target="_blank" rel="noopener noreferrer" 
                                           class="text-blue-600 dark:text-blue-400 hover:underline flex-1 break-all">
                                            {title}
                                        </a>
                                    </div>
                                    """
                                
                                chat_history_html += """
                                        </div>
                                    </details>
                                </div>
                                """
                        except (json.JSONDecodeError, TypeError):
                            pass
                except json.JSONDecodeError:
                    pass
                
                # Continue skipping rendering the raw content for all function responses
                continue
                
            # Handle model messages that are function calls
            if role == "model" and (content.strip().startswith('{"function_call":') or "function_call" in content[:50]):
                try:
                    # Parse the function call JSON
                    fc_data = json.loads(content)
                    fn_name = fc_data.get("function_call", {}).get("name", "")
                    fn_args = fc_data.get("function_call", {}).get("args", {})
                    
                    # Only render code execution function calls
                    if fn_name in ["execute_calculation", "generate_chart", "generate_plotly_chart"]:
                        code = fn_args.get("code", "")
                        if code:
                            # Render as a collapsible code block
                            chat_history_html += f"""
                            <div class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                                <details class="group">
                                    <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                                        <span class="transition group-open:rotate-90">▶</span>
                                        <span>🔧 Code executed via <code class="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">{fn_name}</code></span>
                                    </summary>
                                    <div class="mt-3">
                                        <pre class="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm"><code class="language-python">{code}</code></pre>
                                    </div>
                                </details>
                            </div>
                            """
                    # Skip other function calls (search_web, get_weather, etc.)
                    continue
                except json.JSONDecodeError:
                    # If JSON parsing fails, skip this message
                    continue

            if role == "user":
                chat_history_html += render_user_message(content, image_path)
            else:
                chat_history_html += render_bot_message(content)

    # Fetch previous conversations
    conversations = get_conversations()

    response = templates.TemplateResponse("index.html", {
        "request": request, 
        "chat_history": chat_history_html,
        "conversations": conversations,
        "current_session_id": session_id
    })
    
    # Set cookie if it was missing
    if not request.cookies.get("session_id"):
        response.set_cookie(key="session_id", value=session_id, max_age=31536000) # 1 year
        
    return response

@app.post("/new-chat")
async def new_chat(response: Response):
    """Creates a new chat session."""
    new_session_id = str(uuid.uuid4())
    response = Response()
    response.set_cookie(key="session_id", value=new_session_id, max_age=31536000)
    response.headers["HX-Redirect"] = "/" # Redirect to home to reload sidebar and empty chat
    return response

@app.get("/load-chat/{session_id}")
async def load_chat(session_id: str, response: Response):
    """Loads a specific chat session."""
    response = Response()
    response.set_cookie(key="session_id", value=session_id, max_age=31536000)
    response.headers["HX-Redirect"] = "/" # Redirect to home to reload everything
    return response

# --- 4. Helper Functions for Rendering ---

def render_sidebar_html(current_session_id: str) -> str:
    """Renders the sidebar chat list HTML."""
    conversations = get_conversations()
    sidebar_html = ""
    for chat in conversations:
        is_current = "bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 font-medium" if chat['id'] == current_session_id else "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
        sidebar_html += f"""
        <div class="relative group">
            <button hx-get="/load-chat/{chat['id']}" hx-swap="none"
                class="w-full text-left px-3 py-2 rounded-md text-sm truncate transition-colors flex items-center gap-2 {is_current}">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                <span class="truncate flex-1">{chat['title']}</span>
            </button>
            <!-- Context Menu Button -->
            <button onclick="toggleMenu('menu-{chat['id']}')" 
                class="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-opacity">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500 dark:text-gray-400" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M3 9.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/>
                </svg>
            </button>
            <!-- Dropdown Menu -->
            <div id="menu-{chat['id']}" class="hidden absolute right-0 top-8 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50 min-w-[120px]">
                <button onclick="renameChat('{chat['id']}', '{chat['title']}')" 
                    class="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2">
                    <span>🏷️</span> Rename
                </button>
                <button hx-post="/archive-chat/{chat['id']}" hx-swap="none"
                    class="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2">
                    <span>📦</span> Archive
                </button>
                <button hx-delete="/delete-chat/{chat['id']}" hx-swap="none"
                    hx-confirm="Permanently delete this chat and all associated files?"
                    class="w-full text-left px-3 py-2 text-sm hover:bg-red-50 dark:hover:bg-red-900/20 text-red-600 dark:text-red-400 flex items-center gap-2">
                    <span>🗑️</span> Delete
                </button>
            </div>
        </div>
        """
    return sidebar_html

def render_archived_list_html() -> str:
    """Renders the archived chats list HTML."""
    archived = get_conversations(limit=100, include_archived=True)
    archived = [chat for chat in archived if chat.get('is_archived', 0) == 1]
    
    if not archived:
        return '<div class="text-gray-500 dark:text-gray-400 text-sm">No archived chats</div>'
    
    html = ""
    for chat in archived:
        html += f"""
        <div class="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded-md mb-2">
            <span class="text-sm truncate flex-1 text-gray-700 dark:text-gray-200">{chat['title']}</span>
            <div class="flex gap-2">
                <button hx-post="/unarchive-chat/{chat['id']}" hx-swap="none"
                    class="text-xs px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded">
                    Unarchive
                </button>
                <button hx-delete="/delete-chat/{chat['id']}" hx-swap="none"
                    hx-confirm="Permanently delete this chat and all associated files?"
                    class="text-xs px-2 py-1 bg-red-500 hover:bg-red-600 text-white rounded">
                    Delete
                </button>
            </div>
        </div>
        """
    return html

# --- Routes ---

@app.post("/rename-chat/{session_id}")
async def rename_chat(session_id: str, new_title: str = Form(...)):
    """Rename a chat conversation."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, session_id))
            conn.commit()
        return HTMLResponse("<div>Chat renamed successfully</div>")
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.post("/archive-chat/{session_id}")
async def archive_chat(session_id: str, request: Request):
    """Archive a chat conversation."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET is_archived = 1 WHERE id = ?", (session_id,))
            conn.commit()
        
        current_session_id = request.cookies.get("session_id")
        sidebar_html = render_sidebar_html(current_session_id)
        archived_html = render_archived_list_html()
        
        return HTMLResponse(
            f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>'
            f'<div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>'
        )
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.post("/unarchive-chat/{session_id}")
async def unarchive_chat(session_id: str, request: Request):
    """Unarchive a chat conversation."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET is_archived = 0 WHERE id = ?", (session_id,))
            conn.commit()
            
        current_session_id = request.cookies.get("session_id")
        sidebar_html = render_sidebar_html(current_session_id)
        archived_html = render_archived_list_html()
        
        return HTMLResponse(
            f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>'
            f'<div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>'
        )
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.delete("/delete-chat/{session_id}")
async def delete_chat_route(session_id: str, request: Request):
    """Permanently delete a chat conversation and all associated files."""
    try:
        delete_chat_files(session_id)
        
        current_session_id = request.cookies.get("session_id")
        sidebar_html = render_sidebar_html(current_session_id)
        archived_html = render_archived_list_html()
        
        return HTMLResponse(
            f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>'
            f'<div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>'
        )
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.get("/archived-chats")
async def get_archived_chats():
    """Get list of archived chats for settings modal."""
    try:
        html = render_archived_list_html()
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

# Helper to render messages (DRY)
def render_user_message(content: str, image_path: Optional[str] = None) -> str:
    image_html = ""
    if image_path:
        # Check if it's an image by extension (simple check for rendering)
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            image_html = f'<img src="{image_path}" class="max-h-64 rounded-lg mb-2 border border-blue-400/30">'
        else:
            # Generic file icon
            image_html = f'''
            <div class="flex items-center gap-2 bg-blue-500/20 border border-blue-400/30 rounded-lg p-2 mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 011.414.414l5 5a1 1 0 01.414 1.414V19a2 2 0 01-2 2z" /></svg>
                <span class="text-xs text-white truncate max-w-[200px]">{os.path.basename(image_path)}</span>
            </div>
            '''
        
    return f"""
    <div class="flex justify-end mb-4 animate-fade-in">
        <div class="bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-sm max-w-[80%] shadow-sm">
            {image_html}
            <p class="text-sm leading-relaxed">{content}</p>
        </div>
    </div>
    """

def render_bot_message(content: str, stream_id: Optional[str] = None, final: bool = False, suggestions: List[str] = None) -> str:
    # Extract suggestions from content if not provided
    if suggestions is None:
        import re
        # Look for JSON block at the end
        match = re.search(r'\{\s*"suggestions"\s*:\s*\[(.*?)\]\s*\}\s*$', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                if "suggestions" in data and isinstance(data["suggestions"], list):
                    suggestions = data["suggestions"]
                    # Remove the JSON block from the content to be displayed
                    content = content.replace(json_str, "").strip()
            except json.JSONDecodeError:
                pass

    # If it's a stream placeholder
    if stream_id and not final:
        return f"""
        <div id="{stream_id}" 
             hx-ext="sse" 
             sse-connect="/stream?prompt={urllib.parse.quote(content)}&stream_id={stream_id}" 
             sse-swap="message" 
             class="flex justify-start mb-4 animate-fade-in">
            <div class="bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-gray-800 dark:text-gray-100 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue dark:prose-invert max-w-none">
                <span id="cursor" class="inline-block w-2 h-5 bg-blue-500 cursor-blink align-middle"></span>
            </div>
        </div>
        """
    
    # If it's the final content (or historical content)
    # Preprocess: Convert .html image syntax to links (HTML files can't be embedded as images)
    import re
    content = re.sub(
        r'!\[([^\]]*)\]\(([^)]+\.html)\)',
        r'<a href="\2" onclick="event.stopPropagation(); event.preventDefault(); window.open(this.href, \'_blank\'); return false;" class="text-blue-500 hover:underline">🔗 Open interactive chart</a>',
        content
    )
    
    # Enable tables extension
    html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
    
    suggestions_html = ""
    if suggestions:
        suggestions_html = '<div class="mt-3 flex flex-wrap gap-2">'
        for suggestion in suggestions:
            # Escape quotes in suggestion for onclick attribute
            safe_suggestion = suggestion.replace("'", "\\'")
            suggestions_html += f"""
            <button onclick="setInput('{safe_suggestion}')" 
                class="text-xs bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300 px-3 py-1.5 rounded-full border border-blue-100 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors cursor-pointer">
                {suggestion}
            </button>
            """
        suggestions_html += '</div>'
    
    # If it's an OOB swap update
    if stream_id and final:
        return f"""
        <div id="{stream_id}" 
             hx-swap-oob="outerHTML:#{stream_id}" 
             class="flex justify-start mb-4">
            <div class="bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-gray-800 dark:text-gray-100 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue dark:prose-invert max-w-none">
                {html_content}
                {suggestions_html}
            </div>
        </div>
        """
    
    # Just static HTML (for history)
    return f"""
    <div class="flex justify-start mb-4">
        <div class="bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-gray-800 dark:text-gray-100 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue dark:prose-invert max-w-none">
            {html_content}
            {suggestions_html}
        </div>
    </div>
    """

# Helper to format SSE strings
def format_sse(data: str, event: str = "message") -> str:
    msg = f"event: {event}\n"
    for line in data.splitlines():
        msg += f"data: {line}\n"
    msg += "\n"
    return msg

# --- 5. TOOLS ---
def import_package(package_name,install=True):
    """
    Checks if a package is installed. If not, installs it via pip
    and then imports it.
    Use this tool when you need to use a specific package, that is not installed to solve a problem involving the execution of code.
    Use this tool also when you only want to check if a package is installed, but for a good reason you do not want to install when it's not installed (install=False).
    Args:
        package_name (str): The name of the package on PyPI (e.g., "PyYAML").
        install (bool): Whether to install the package if it is not installed. Default is True.

    Returns:
        module: The imported module object (or errormessage if failed to install)
    """
    try:
        print(f"Importing {package_name}")
        res = importlib.import_module(package_name)
        return {"response": "Successfully imported {package_name}"}
    except ImportError:
        try:
            if install:
                print(f"Failed to import {package_name}, installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])            
                res = importlib.import_module(package_name)
                importlib.invalidate_caches()
                return {"response": "Succesfully installed & imported {package_name}: {e}"}
            else:
                print(f"Failed to import {package_name}, NOT installing...")
                return {"response": "Failed to import {package_name}: {e}"}
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {e}")
            return {"response": "Failed to install {package_name}: {e}"}

def write_source_code(file_path: str, code: str):
    """
    Writes the file containing the source code to the specified path.
    Use this tool when you need to write the source code of a file.
    Args:
        file_path (str): The path to the file to write.This may NOT be "main.py", if not an error message is returned.
        code (str): String containing the modified source code to write to the file.

    Returns:
        str: success or error message
    """
    try:
        if file_path!="main.py":
            with open(file_path, 'w') as f:
                print(f"Writing file {file_path}")
                f.write(code)
                return f"Successfully wrote file {file_path}"
        else:
            return f"Target file name not allowed: {file_path}"    
    except Exception as e:
        return f"Error writing file {file_path}: {e}"  

def read_source_code(file_path: str):
    """
    Reads the contents of a source code file and returns it as a string.
    Use this tool when you need to read the source code of a file.
    Args:
        file_path (str): The path to the file to read. This may NOT be "main.py", if not an error message is returned.

    Returns:
        str: String with the entire contents of the source code.
    """
    try:
        if file_path!="main.py":
            with open(file_path, 'r') as f:
                print(f"Reading file {file_path}")
                code = f.read()
                return code
        else:
            return f"Source code file name not allowed: {file_path}"   
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

  

def search_web(query: str):
    """
    Searches the web for the given query using Tavily to get up-to-date information.
    Use this tool when the user asks about current events, news, or specific information 
    that might not be in your training data.
    """
    if not tavily_client:
        return "Error: Web search is not configured (missing API key)."
    
    try:
        print(f"Searching web for: {query}")
        response = tavily_client.search(query=query, search_depth="basic")
        results = response.get('results', [])[:3]  # Limit to top 3
        
        # Format results for the model
        context = "\n".join([
            f"Source: {result['url']}\nContent: {result['content']}" 
            for result in results
        ])
        
        # Return JSON with both context and sources for display
        return json.dumps({
            "context": context,
            "sources": [{"url": r['url'], "title": r.get('title', r['url'])} for r in results]
        })
    except Exception as e:
        return f"Error performing search: {str(e)}"


def crawl_website(url: str, max_depth: int = 2, limit: int = 10, instructions: str = None):
    """
    Crawls a website starting from the given URL to extract content from multiple pages.
    Use this tool when you need to gather information from multiple pages of a website,
    or when you need to explore a website's structure and content.
    
    Args:
        url (str): The starting URL to crawl from.
        max_depth (int): How deep to crawl (number of link levels to follow). Default is 2.
        limit (int): Maximum number of pages to crawl. Default is 10.
        instructions (str): Optional instructions to filter which pages to include (e.g., "Find all pages about pricing").
    
    Returns:
        str: JSON with crawled content and sources.
    """
    if not tavily_client:
        return "Error: Web crawl is not configured (missing API key)."
    
    try:
        print(f"Crawling website: {url} (depth={max_depth}, limit={limit})")
        
        # Build crawl parameters
        crawl_params = {
            "url": url,
            "max_depth": max_depth,
            "limit": limit
        }
        if instructions:
            crawl_params["instructions"] = instructions
        
        response = tavily_client.crawl(**crawl_params)
        results = response.get('results', [])
        
        # Format results for the model
        context_parts = []
        sources = []
        for result in results:
            page_url = result.get('url', url)
            raw_content = result.get('raw_content', '')[:2000]  # Limit content per page
            context_parts.append(f"Page: {page_url}\nContent: {raw_content}")
            sources.append({"url": page_url, "title": result.get('title', page_url)})
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Return JSON with both context and sources for display
        return json.dumps({
            "context": context,
            "sources": sources,
            "pages_crawled": len(results)
        })
    except Exception as e:
        return f"Error crawling website: {str(e)}"

def execute_calculation(code: str):
    """
    Executes Python code for calculations, logic, and text processing.
    Use this for math, data analysis, or string manipulation.
    
    IMPORTANT:
    - This tool does NOT support plotting or image generation. Use 'generate_chart' for that.
    - The code must print the final result to stdout using `print()`.
    - You can import standard libraries (math, datetime, json, etc.) and numpy/pandas.
    """
    # Create a captured output stream
    output_capture = io.StringIO()
    
    try:
        # Redirect stdout to capture print statements
        with contextlib.redirect_stdout(output_capture):
            # Execute the code in a restricted global scope
            # We allow numpy and pandas for data analysis
            exec(code, {'__builtins__': __builtins__, 'np': np, 'pd': pd, 'plt': plt})
            
        result = output_capture.getvalue()
        if not result:
            return "Code executed successfully but printed no output. Did you forget to print the result?"
        return result.strip()
        
    except Exception as e:
        return f"Error executing code: {str(e)}"

def generate_chart(code: str):
    """
    Generates a chart or plot using Python and matplotlib.
    Use this tool WHENEVER the user asks for a visualization, graph, or chart.
    
    IMPORTANT:
    - You MUST use `plt.savefig('plot.png')` (or any filename ending in .png) to save the plot.
    - Do NOT use `plt.show()`.
    - The code should clear the figure before plotting: `plt.clf()` or `plt.close()`.
    - The tool will return the path to the generated image.
    """
    # Ensure generated directory exists
    os.makedirs(GENERATED_DIR, exist_ok=True)
    
    # Create a captured output stream
    output_capture = io.StringIO()
    
    try:
        # Redirect stdout
        with contextlib.redirect_stdout(output_capture):
            # Execute the code
            # Ensure we are in a clean state
            plt.clf()
            exec(code, {'__builtins__': __builtins__, 'np': np, 'pd': pd, 'plt': plt})
            
        # Check for generated images in the current directory (which is the root)
        # We look for recently created .png files
        generated_file = None
        
        # First, check if any PNG files were created in current directory
        for file in os.listdir('.'):
            if file.endswith('.png'):
                # Move to static/generated
                new_filename = f"chart_{uuid.uuid4()}.png"
                dest_path = os.path.join(GENERATED_DIR, new_filename)
                shutil.move(file, dest_path)
                generated_file = f"/static/generated/{new_filename}"
                print(f"Found and moved chart: {file} -> {dest_path}")
                break
        
        # If no file was found, try to save the current figure automatically
        if not generated_file:
            print("No PNG file found. Attempting to auto-save current figure...")
            new_filename = f"chart_{uuid.uuid4()}.png"
            dest_path = os.path.join(GENERATED_DIR, new_filename)
            plt.savefig(dest_path, bbox_inches='tight', dpi=100)
            generated_file = f"/static/generated/{new_filename}"
            print(f"Auto-saved chart to: {dest_path}")
        
        output = output_capture.getvalue().strip()
        
        if generated_file:
            # Save to database if session_id is available
            session_id = session_id_ctx.get()
            if session_id:
                try:
                    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                        # Store the absolute path for deletion purposes, or relative if preferred.
                        # delete_chat_files uses os.remove(file_path), so we need the filesystem path.
                        # generated_file is the web path (/static/generated/...).
                        # dest_path is the filesystem path.
                        conn.execute("INSERT INTO generated_files (session_id, file_path) VALUES (?, ?)", (session_id, dest_path))
                        print(f"Saved generated file record for session {session_id}: {dest_path}")
                except Exception as db_e:
                    print(f"Error saving generated file to DB: {db_e}")

            return json.dumps({
                "status": "success",
                "output": output,
                "image_path": generated_file,
                "message": "Chart generated successfully. I will now analyze it."
            })
        else:
            return f"Code executed but no chart was saved. Did you forget `plt.savefig(...)`? Output: {output}"
        
    except Exception as e:
        print(f"Error in generate_chart: {str(e)}")
        traceback.print_exc()
        return f"Error generating chart: {str(e)}"


def generate_plotly_chart(code: str):
    """
    Generates an advanced chart using Python and Plotly.
    Use this tool for complex, interactive, or 3D visualizations that matplotlib cannot handle well.
    
    Examples of when to use this over generate_chart:
    - 3D plots and surfaces
    - Interactive charts (hover, zoom, pan)
    - Sunburst/treemap charts
    - Sankey diagrams
    - Animated charts
    - Geographic/map visualizations
    
    IMPORTANT:
    - You MUST assign the figure to a variable named `fig`.
    - The tool will automatically save `fig` as both a static PNG (for preview) and an interactive HTML file.
    - You do NOT need to call `fig.write_image` or `fig.write_html` yourself, but you can if you want specific filenames.
    - Use `import plotly.express as px` or `import plotly.graph_objects as go`.
    """
    # Ensure generated directory exists
    os.makedirs(GENERATED_DIR, exist_ok=True)
    
    # Create a captured output stream
    output_capture = io.StringIO()
    
    try:
        # Redirect stdout
        with contextlib.redirect_stdout(output_capture):
            # Execute the code
            local_vars = {
                '__builtins__': __builtins__, 
                'np': np, 
                'pd': pd, 
                'px': px, 
                'go': go,
                'pio': pio
            }
            exec(code, local_vars)
            
        # Check for generated files (PNG or HTML) created manually by the code
        generated_png = None
        generated_html = None
        
        # Also check if 'fig' exists in locals and auto-save if needed
        fig = local_vars.get('fig')
        if fig:
            # Auto-save logic
            base_uuid = str(uuid.uuid4())
            
            # Save PNG
            png_filename = f"plotly_chart_{base_uuid}.png"
            png_dest = os.path.join(GENERATED_DIR, png_filename)
            fig.write_image(png_dest)
            generated_png = f"/static/generated/{png_filename}"
            print(f"Auto-saved Plotly PNG: {png_dest}")
            
            # Save HTML
            html_filename = f"plotly_chart_{base_uuid}.html"
            html_dest = os.path.join(GENERATED_DIR, html_filename)
            fig.write_html(html_dest)
            generated_html = f"/static/generated/{html_filename}"
            print(f"Auto-saved Plotly HTML: {html_dest}")
            
        else:
            # Fallback: check if files were created manually in current directory
            for file in os.listdir('.'):
                if file.endswith('.png') and 'plotly' in file: # loose check
                     # Move to static/generated
                    new_filename = f"plotly_chart_{uuid.uuid4()}.png"
                    dest_path = os.path.join(GENERATED_DIR, new_filename)
                    shutil.move(file, dest_path)
                    generated_png = f"/static/generated/{new_filename}"
                    
                elif file.endswith('.html') and 'plotly' in file:
                    new_filename = f"plotly_chart_{uuid.uuid4()}.html"
                    dest_path = os.path.join(GENERATED_DIR, new_filename)
                    shutil.move(file, dest_path)
                    generated_html = f"/static/generated/{new_filename}"

        output = output_capture.getvalue().strip()
        
        if generated_png or generated_html:
            # Save to database if session_id is available
            session_id = session_id_ctx.get()
            if session_id:
                try:
                    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                        if generated_png:
                            conn.execute("INSERT INTO generated_files (session_id, file_path) VALUES (?, ?)", (session_id, os.path.join(GENERATED_DIR, os.path.basename(generated_png))))
                        if generated_html:
                            conn.execute("INSERT INTO generated_files (session_id, file_path) VALUES (?, ?)", (session_id, os.path.join(GENERATED_DIR, os.path.basename(generated_html))))
                except Exception as db_e:
                    print(f"Error saving generated file to DB: {db_e}")

            # Construct response
            response_data = {
                "status": "success",
                "output": output,
                "message": "Plotly chart generated successfully."
            }
            
            if generated_png:
                response_data["image_path"] = generated_png
            
            if generated_html:
                response_data["html_path"] = generated_html
                # Add the link message
                link_msg = f'Interactive Plotly chart generated. <a href="{generated_html}" onclick="event.stopPropagation(); event.preventDefault(); window.open(this.href, \'_blank\'); return false;" class="text-blue-500 hover:underline">🔗 Open chart in new tab</a>'
                response_data["message"] = link_msg

            return json.dumps(response_data)
        else:
            return f"Code executed but no chart was saved. Did you assign the figure to `fig`? Output: {output}"
        
    except Exception as e:
        print(f"Error in generate_plotly_chart: {str(e)}")
        traceback.print_exc()
        return f"Error generating Plotly chart: {str(e)}"

def wolfram_alpha_query(query: str):
    """
    Queries the Wolfram Alpha LLM API for scientific, mathematical, or factual information.
    Use this tool for:
    - Complex math calculations (integrals, derivatives, solving equations)
    - Scientific data (chemistry, physics, astronomy)
    - Unit conversions and physical constants
    - Factual queries about geography, history, etc.
    
    The query should be a single-line natural language string or math expression.
    """
    app_id = os.getenv("WOLFRAM_ALPHA_APP_ID")
    if not app_id:
        return "Error: WOLFRAM_ALPHA_APP_ID not found in environment variables."
        
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    
    try:
        params = {
            "appid": app_id,
            "input": query
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        print(f"Querying Wolfram Alpha: {url}")
        
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
            
    except Exception as e:
        print(f"Error querying Wolfram Alpha: {str(e)}")
        return f"Error querying Wolfram Alpha: {str(e)}"

def get_timezone_from_ip(ip_address: str) -> Optional[str]:
    """
    Fetches the timezone for a given IP address using ip-api.com.
    Returns None if lookup fails.
    """
    try:
        # Handle local IP
        if ip_address in ("127.0.0.1", "::1", "localhost"):
            return None # Use system default or UTC
            
        url = f"http://ip-api.com/json/{ip_address}?fields=timezone"
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read().decode())
            if "timezone" in data:
                return data["timezone"]
    except Exception as e:
        print(f"Error fetching timezone for IP {ip_address}: {e}")
    return None

def get_user_timezone():
    """
    Returns the timezone of the user based on their IP address.
    Use this tool when the user asks "What is my timezone?" or similar questions.
    """
    client_ip = client_ip_ctx.get()
    if not client_ip:
        return "Error: Could not determine client IP."
        
    tz = get_timezone_from_ip(client_ip)
    if tz:
        return f"User Timezone: {tz}"
    else:
        return f"Could not determine timezone for IP: {client_ip}"

def get_current_datetime(timezone: str = None):
    """
    Returns the current date and time.
    Args:
        timezone (str, optional): The timezone to use (e.g., 'America/New_York', 'Asia/Tokyo').
                                  If not provided, tries to detect from IP, otherwise defaults to UTC.
    """
    tz = None
    
    # 1. Try user-provided timezone
    if timezone:
        try:
            if ZoneInfo:
                tz = ZoneInfo(timezone)
            else:
                print("ZoneInfo not available, ignoring timezone arg")
        except Exception as e:
            return f"Error: Invalid timezone '{timezone}'. Please use a valid IANA timezone name (e.g., 'Europe/London')."

    # 2. Try IP detection if no timezone provided
    if not tz:
        client_ip = client_ip_ctx.get()
        if client_ip:
            detected_tz_name = get_timezone_from_ip(client_ip)
            if detected_tz_name and ZoneInfo:
                try:
                    tz = ZoneInfo(detected_tz_name)
                    print(f"Detected timezone from IP {client_ip}: {detected_tz_name}")
                except Exception:
                    pass
    
    # 3. Get time
    if tz:
        now = datetime.datetime.now(tz)
        tz_name = str(tz)
    else:
        now = datetime.datetime.now(datetime.timezone.utc)
        tz_name = "UTC"
        
    return f"Current Date and Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"

def generate_image(description: str):
    """
    Generates an artistic or creative image based on the description using AI.
    Use this tool for artistic requests like "draw a cat", "create a sunset landscape", etc.
    DO NOT use this for data visualizations, charts, or graphs - use generate_chart instead.
    
    Args:
        description (str): A detailed description of the image to generate.
    
    Returns:
        str: Markdown formatted image that will be displayed to the user.
    """
    # URL encode the description
    import urllib.parse
    encoded_description = urllib.parse.quote(description)
    
    # Generate Pollinations.ai URL
    image_url = f"https://image.pollinations.ai/prompt/{encoded_description}?width=1024&height=1024&nologo=true"
    
    # Return markdown image
    return f"![{description}]({image_url})"

def get_coordinates(location: str):
    """
    Returns the latitude and longitude of a specific location (city, landmark, address).
    Use this tool when the user asks for coordinates or location data.
    """
    try:
        geolocator = Nominatim(user_agent="htmx_chatbot")
        loc = geolocator.geocode(location)
        if loc:
            return f"Location: {loc.address}\nLatitude: {loc.latitude}\nLongitude: {loc.longitude}"
        else:
            return f"Error: Could not find coordinates for '{location}'."
    except Exception as e:
        return f"Error getting coordinates: {str(e)}"

def get_weather(location: str):
    """
    Gets current weather information for a specific location using XWeather MCP server.
    Use this tool when the user asks about weather, temperature, conditions,  etc.
    
    Args:
        location (str): The location to get weather for (city name, address, etc.)
    
    Returns:
        str: Current weather information for the location
    """
    try:
        import requests
        
        # Get XWeather credentials from environment
        xweather_id = os.getenv("XWEATHER_MCP_ID")
        xweather_secret = os.getenv("XWEATHER_MCP_SECRET")
        
        if not xweather_id or not xweather_secret:
            return "Error: XWeather API credentials not configured. Please set XWEATHER_MCP_ID and XWEATHER_MCP_SECRET in .env file."
        
        # MCP server endpoint
        mcp_url = "https://mcp.api.xweather.com/mcp"
        
        # Prepare authorization (using client_id:client_secret format)
        auth_string = f"{xweather_id}_{xweather_secret}"
        
        # Make request to MCP server
        # The MCP server expects a JSON-RPC 2.0 request
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "xweather_get_current_weather",
                "arguments": {
                    "location": location
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {auth_string}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        response = requests.post(mcp_url, json=payload, headers=headers, timeout=10)
        
        print(f"XWeather API Status: {response.status_code}")
        print(f"XWeather API Response: {response.text[:500]}")  # Print first 500 chars
        
        if response.status_code == 200:
            try:
                # The response is in SSE format, need to parse it
                response_text = response.text
                
                # Extract JSON from SSE data field
                if "data:" in response_text:
                    # Split by lines and find the data line
                    for line in response_text.split('\n'):
                        if line.startswith('data:'):
                            json_str = line[5:].strip()  # Remove 'data:' prefix
                            data = json.loads(json_str)
                            
                            # Check for errors in the result
                            if "result" in data:
                                result = data["result"]
                                if isinstance(result, dict) and result.get("isError"):
                                    # Extract error message
                                    if "content" in result and len(result["content"]) > 0:
                                        error_text = result["content"][0].get("text", "Unknown error")
                                        # Add helpful hint for invalid location errors
                                        if "invalid" in error_text.lower() and "location" in error_text.lower():
                                            error_text += "\n\nHint: Try specifying the location more precisely, e.g., 'Minneapolis, MN', 'London, UK', or use a zip code like '55401'."
                                        return f"Weather service error: {error_text}"
                                    return f"Weather service error: {result}"
                                
                                # Return successful result
                                if "content" in result and len(result["content"]) > 0:
                                    return result["content"][0].get("text", str(result))
                                return str(result)
                            else:
                                return str(data)
                    
                    return f"Error: Could not parse SSE response: {response_text[:200]}"
                else:
                    # Try parsing as regular JSON
                    data = response.json()
                    if "result" in data:
                        return str(data["result"])
                    else:
                        return str(data)
                        
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response content: {response.text}")
                return f"Error: Received invalid JSON from XWeather API. Response: {response.text[:200]}"
        else:
            print(response.text)
            return f"Error: XWeather API returned status {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error getting weather: {str(e)}"

def get_forecast_weather(location: str):
    """
    Gets multi-day weather forecast for a specific location using XWeather MCP server.
    Use this tool for planning, event scheduling, travel preparation, or when the user asks about future weather.
    
    Args:
        location (str): The location to get forecast for (city, state/country, or zip code)
    
    Returns:
        str: Multi-day weather forecast with temperatures, precipitation, wind, and conditions
    """
    try:
        import requests
        
        xweather_id = os.getenv("XWEATHER_MCP_ID")
        xweather_secret = os.getenv("XWEATHER_MCP_SECRET")
        
        if not xweather_id or not xweather_secret:
            return "Error: XWeather API credentials not configured."
        
        mcp_url = "https://mcp.api.xweather.com/mcp"
        auth_string = f"{xweather_id}_{xweather_secret}"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "xweather_get_forecast_weather",
                "arguments": {
                    "location": location
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {auth_string}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        response = requests.post(mcp_url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            response_text = response.text
            if "data:" in response_text:
                for line in response_text.split('\n'):
                    if line.startswith('data:'):
                        json_str = line[5:].strip()
                        try:
                            data = json.loads(json_str)
                            
                            if "result" in data:
                                result = data["result"]
                                if isinstance(result, dict) and result.get("isError"):
                                    if "content" in result and len(result["content"]) > 0:
                                        error_text = result["content"][0].get("text", "Unknown error")
                                        if "invalid" in error_text.lower() and "location" in error_text.lower():
                                            error_text += "\n\nHint: Try specifying the location more precisely, e.g., 'Minneapolis, MN', 'London, UK', or use a zip code."
                                        return f"Weather service error: {error_text}"
                                
                                if "content" in result and len(result["content"]) > 0:
                                    return result["content"][0].get("text", str(result))
                                return str(result)
                        except json.JSONDecodeError:
                            pass
                            
                return f"Error: Could not parse forecast response"
        else:
            return f"Error: XWeather API returned status {response.status_code}"
            
    except Exception as e:
        return f"Error getting forecast: {str(e)}"

def get_precipitation_timing(location: str):
    """
    Gets timing information for upcoming precipitation (rain, snow, etc.) in the next hours to days.
    Use this tool when the user asks "when will it rain?", "when will it stop raining?", or needs precipitation timing for outdoor activities.
    
    Args:
        location (str): The location to get precipitation timing for (city, state/country, or zip code)
    
    Returns:
        str: Start/stop timing and duration of upcoming precipitation
    """
    try:
        import requests
        
        xweather_id = os.getenv("XWEATHER_MCP_ID")
        xweather_secret = os.getenv("XWEATHER_MCP_SECRET")
        
        if not xweather_id or not xweather_secret:
            return "Error: XWeather API credentials not configured."
        
        mcp_url = "https://mcp.api.xweather.com/mcp"
        auth_string = f"{xweather_id}_{xweather_secret}"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "xweather_get_forecast_precipitation_timing",
                "arguments": {
                    "location": location
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {auth_string}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        response = requests.post(mcp_url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            response_text = response.text
            if "data:" in response_text:
                for line in response_text.split('\n'):
                    if line.startswith('data:'):
                        json_str = line[5:].strip()
                        data = json.loads(json_str)
                        
                        if "result" in data:
                            result = data["result"]
                            if isinstance(result, dict) and result.get("isError"):
                                if "content" in result and len(result["content"]) > 0:
                                    error_text = result["content"][0].get("text", "Unknown error")
                                    if "invalid" in error_text.lower() and "location" in error_text.lower():
                                        error_text += "\n\nHint: Try specifying the location more precisely, e.g., 'Minneapolis, MN', 'London, UK', or use a zip code."
                                    return f"Weather service error: {error_text}"
                            
                            if "content" in result and len(result["content"]) > 0:
                                return result["content"][0].get("text", str(result))
                            return str(result)
                        
                return f"Error: Could not parse precipitation timing response"
        else:
            return f"Error: XWeather API returned status {response.status_code}"
            
    except Exception as e:
        return f"Error getting precipitation timing: {str(e)}"

# --- 4. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request, response: Response):
    """
    On first load, check for a cookie. If missing, generate one.
    """
    # Check if user already has a session cookie
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        session_id = str(uuid.uuid4())
        

@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, prompt: str = Form(...), file: UploadFile = File(None)):
    
    session_id = request.cookies.get("session_id")
    image_path = None
    mime_type = None
    
    if file and file.filename:
        # Save file
        os.makedirs("static/uploads", exist_ok=True)
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = f"static/uploads/{filename}"
        with open(filepath, "wb") as f:
            f.write(await file.read())
        image_path = f"/{filepath}" # Web path
        mime_type = file.content_type
    
    # Save User Message
    if session_id:
        save_message(session_id, "user", prompt, image_path, mime_type)

    stream_id = f"msg-{uuid.uuid4()}"
    
    # Render User Message
    user_message_html = render_user_message(prompt, image_path)
    
    # Render Bot Placeholder
    # Pass image_path in URL for stream
    # URL encode the prompt to ensure valid HTML attributes and query strings
    safe_prompt = urllib.parse.quote(prompt)
    stream_url = f"/stream?prompt={safe_prompt}&stream_id={stream_id}"
    if image_path:
        stream_url += f"&image_path={image_path}"
    if mime_type:
        stream_url += f"&mime_type={urllib.parse.quote(mime_type)}"
        
    # We need to manually construct the placeholder since render_bot_message doesn't support custom URL yet
    # Or we update render_bot_message. Let's update render_bot_message signature in previous chunk? 
    # No, let's just construct it here to be safe and explicit.
    bot_placeholder_html = f"""
    <div id="{stream_id}" 
         hx-ext="sse" 
         sse-connect="{stream_url}" 
         sse-swap="message" 
         class="flex justify-start mb-4 animate-fade-in">
        <div class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
            <span id="cursor" class="inline-block w-2 h-5 bg-blue-500 cursor-blink align-middle"></span>
        </div>
    </div>
    """
    
    return user_message_html + bot_placeholder_html

@app.get("/stream")
async def stream_response(request: Request, prompt: str, session_id: str = Cookie(None), stream_id: str = Query(...), image_path: Optional[str] = None, mime_type: Optional[str] = None, selected_model: str = Cookie("gemini-2.5-flash")):
    """
    The 'session_id' is automatically extracted from the browser cookie.
    """
    # Set client IP in context for tools to use
    if request.client:
        client_ip_ctx.set(request.client.host)
    
    if session_id:
        session_id_ctx.set(session_id)
        
    async def event_generator():
        error_div = '<div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm"><strong class="font-semibold">Error:</strong> An unexpected error occurred. Please try again.</div>'
        try:
            # If no session cookie, we can't track history.
            if not session_id:
                yield format_sse("Error: No session found. Refresh page.")
                return

            # Prepare messages payload
            messages_payload = []
            
            # Add history
            if session_id:
                messages_payload.extend(get_history(session_id))
                
            # Add current message
            current_parts = [prompt]
            uploaded_file = None # Initialize uploaded_file
            
            if image_path:
                # Check if it is an image or generic file
                # For this demo, we assume the file is on local disk at relative path
                # image_path comes as "/static/uploads/..."
                local_path = image_path.lstrip("/")
                
                try:
                    # Upload to Gemini
                    print(f"Uploading file to Gemini: {local_path} ({mime_type})")
                    uploaded_file = genai.upload_file(local_path, mime_type=mime_type)
                    
                    # Wait for processing if needed (mostly for videos, but good practice)
                    while uploaded_file.state.name == "PROCESSING":
                        print("Waiting for file processing...")
                        await asyncio.sleep(1)
                        uploaded_file = genai.get_file(uploaded_file.name)
                        
                    if uploaded_file.state.name == "FAILED":
                        yield format_sse("<div><strong>Error:</strong> File processing failed.</div>")
                        return

                    current_parts.append(uploaded_file)
                    print(f"File uploaded successfully: {uploaded_file.uri}")
                    
                    # Update the message in DB with the Gemini URI for future context
                    # We need to find the message ID. Since we don't have it easily (it was just inserted),
                    # we can update by session_id and image_path (assuming unique enough for this flow)
                    # or better: update the LAST message for this session that has this image_path
                    if session_id:
                        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                            conn.execute(
                                """
                                UPDATE messages 
                                SET gemini_uri = ? 
                                WHERE id = (
                                    SELECT id FROM messages 
                                    WHERE session_id = ? AND image_path = ? 
                                    ORDER BY id DESC LIMIT 1
                                )
                                """,
                                (uploaded_file.uri, session_id, image_path)
                            )
                    
                except Exception as e:
                    print(f"Error uploading file: {e}")
                    yield format_sse(f"<div><strong>Error uploading file:</strong> {str(e)}</div>")
                    # Continue without file? Or return? Let's return to be safe.
                    return

            messages_payload.append({
                "role": "user",
                "parts": current_parts
            })

            # 2. Call Gemini
            # Prepare the model's tools and system instruction
            tools = []
            current_instruction = get_system_instruction() # Start with base instruction
            has_file_in_context = uploaded_file is not None # Check if file was uploaded in this turn
            
            # Check history for files too
            if not has_file_in_context:
                for msg in messages_payload:
                    for part in msg.get('parts', []):
                        if hasattr(part, 'file_data') and part.file_data:
                            has_file_in_context = True
                            break
                    if has_file_in_context:
                        break

            if not has_file_in_context:
                if tavily_client:
                    tools.append(search_web)
                    current_instruction += "\n\n- You have access to a 'search_web' tool. You must use it whenever the user asks for current information, news, or facts you don't know. Do NOT invent new tools. Only use 'search_web' for searching."
                    
                    tools.append(crawl_website)
                    current_instruction += "\n\n- You have access to a 'crawl_website' tool. Use it when you need to gather content from multiple pages of a specific website, explore a site's structure, or extract information from specific URLs. Parameters: url (required), max_depth (default 2), limit (default 10), instructions (optional filter)."
                
                tools.append(read_source_code)
                current_instruction += "\n\n- You have access to a 'read_source_code' tool. Use it for reading your own source code files. It returns TEXT output. It CANNOT generate images."
                
                tools.append(write_source_code)
                current_instruction += "\n\n- You have access to a 'write_source_code' tool. Use it for writing (modified) source code to file. It returns TEXT output. It CANNOT generate images."

                tools.append(import_package)
                current_instruction += "\n\n- You have access to an 'import_package' tool. Use it for checking if a package is installed (install=False) or installing and importing directly (install=True). It returns TEXT output. It CANNOT generate images. It can also just check if a package is installed"
                
                tools.append(execute_calculation)
                current_instruction += "\n\n- You have access to an 'execute_calculation' tool. Use it for math, logic, text processing, or data analysis (numpy/pandas) or any arbitray python code executions. It returns TEXT output. It CANNOT generate images."

                tools.append(generate_chart)
                current_instruction += "\n\n- You have access to a 'generate_chart' tool using matplotlib. Use this for standard charts (bar, line, pie, scatter, histograms). DO NOT provide Python code to the user - ALWAYS call the tool."

                tools.append(generate_plotly_chart)
                current_instruction += "\n\n- You have access to a 'generate_plotly_chart' tool using Plotly. Use this for ADVANCED visualizations: 3D plots, interactive charts, sunburst/treemap, Sankey diagrams, animated charts, geographic maps. Use this when matplotlib's generate_chart cannot handle the request."

                tools.append(wolfram_alpha_query)
                current_instruction += """\n\n- You have access to a 'wolfram_alpha_query' tool. Use it for complex math, scientific data, unit conversions, and factual queries.
GUIDELINES:
- Convert inputs to simplified keyword queries (e.g. "France population" instead of "how many people live in France").
- Send queries in English only.
- Use proper Markdown for math formulas: '$$...$$' for block, '\\(...\\)' for inline.
- Use named physical constants (e.g., 'speed of light') without numerical substitution.
- Include a space between compound units (e.g., "Ω m").
- If data for multiple properties is needed, make separate calls for each property.
- If the result is not relevant, try re-sending with 'Assumption' parameters if suggested by Wolfram."""

                tools.append(generate_image)
                current_instruction += "\n\n- You have access to a 'generate_image' tool. Use it for artistic or creative image requests like 'draw a cat', 'create a sunset landscape', etc. DO NOT use this for data visualizations - use generate_chart instead."

                tools.append(get_current_datetime)
                current_instruction += "\n\n- You have access to a 'get_current_datetime' tool. Use it when asked about the current time or date. It automatically detects the user's timezone from their IP. You can also pass a 'timezone' argument (e.g., 'Asia/Tokyo') if the user explicitly requests a specific timezone."

                tools.append(get_user_timezone)
                current_instruction += "\n\n- You have access to a 'get_user_timezone' tool. Use it when the user asks 'What is my timezone?' or wants to know the timezone detected from their IP."

                tools.append(get_coordinates)
                current_instruction += "\n\n- You have access to a 'get_coordinates' tool. Use it to find the latitude and longitude of locations."

                tools.append(get_weather)
                current_instruction += "\n\n- You have access to a 'get_weather' tool. Use it when the user asks about current weather, temperature, or conditions for any location."

                tools.append(get_forecast_weather)
                current_instruction += "\n\n- You have access to a 'get_forecast_weather' tool. Use it for multi-day weather forecasts, planning, event scheduling, or when the user asks about future weather."

                tools.append(get_precipitation_timing)
                current_instruction += "\n\n- You have access to a 'get_precipitation_timing' tool. Use it when the user asks 'when will it rain?', 'when will it stop raining?', or needs precipitation timing information."
            else:
                # When files are present, clear history to avoid API compatibility issues
                # Keep only the current message (last one in payload)
                print("Tools disabled due to file context. Clearing history to avoid API errors.")
                if messages_payload:
                    current_msg = messages_payload[-1]  # Keep only the current message
                    messages_payload = [current_msg]
                
                current_instruction += "\n\nNote: Tool calling is disabled because you are analyzing an uploaded file. Focus on the file content."
            
            current_instruction += """\n\n- if you have actionable follow-up suggestions, append them to the very end of your response as a JSON object with the key 'suggestions', like this: {"suggestions": ["Action 1", "Action 2"]}. Do not wrap this in markdown code blocks. Make sure it is the last thing in your response."""
            current_instruction += """\n\n- when providing a URL make sure it's clickable, with target being a new tab"""
            current_instruction += """\n\n- when using search_web tool, remember the urls that were used"""
            
            # Debug: Print registered tools
            tool_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in tools]
            print(f"Registered tools: {tool_names}")

            model = genai.GenerativeModel(
                model_name=selected_model,
                system_instruction=current_instruction,
                tools=tools if tools else None
            )
            
            # We need to handle potential function calls in a loop
            # Since we are streaming, this is a bit tricky with the official SDK's generate_content(stream=True)
            # automatic function calling isn't fully supported in stream mode for single-turn logic easily without chat session.
            # But we can do it manually.
            
            # Auto-continue loop - will retry if model says "please wait" etc.
            auto_continue_count = 0
            max_auto_continues = 3
            should_auto_continue = True
            
            while should_auto_continue and auto_continue_count <= max_auto_continues:
                should_auto_continue = False  # Reset flag, will be set if retry phrase detected
                
                # Main loop for handling function calls
                turn = 0
                max_turns = 10
                full_response_text = ""  # Initialize to prevent UnboundLocalError
            
                while turn < max_turns:
                    turn += 1
                    # First attempt - Use stream=False to safely check for function calls
                    # This avoids complexity with iterating streams for tool use
                    print("Sending request to model (stream=False)...")
                    if turn > 1:
                         yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Thinking (Step {turn})...</div>')
                    else:
                         yield format_sse('<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Thinking...</div>')
                    
                    response = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = model.generate_content(messages_payload, stream=False)
                            # Check if valid
                            if response.candidates and response.candidates[0].content.parts:
                                break # Success
                            else:
                                print(f"Attempt {attempt+1}/{max_retries}: Empty response. Retrying...")
                                yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Thinking (Attempt {attempt+1})...</div>')
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(0.5)
                        except ResourceExhausted as e:
                            print(f"Quota exceeded: {e}")
                            yield format_sse(f'<div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm"><strong class="font-semibold">Quota Exceeded:</strong> You have hit the API rate limit. Please wait a moment and try again.</div>')
                            return # Stop processing immediately
                        except Exception as e:
                            error_str = str(e)
                            
                            # Check for token limit errors (don't retry)
                            if "token count exceeds" in error_str.lower() or "maximum number of tokens" in error_str.lower():
                                print(f"Token limit exceeded: {e}")
                                yield format_sse(f'<div class="bg-orange-50 border border-orange-200 text-orange-700 px-4 py-3 rounded-lg text-sm"><strong class="font-semibold">Token Limit Exceeded:</strong> The input (including file content and chat history) is too large. The maximum allowed is 1,048,576 tokens. Please try with a smaller file or clear your chat history.</div>')
                                return # Stop processing immediately
                            
                            # Check for invalid argument errors (don't retry)
                            if "400" in error_str and "invalid argument" in error_str.lower():
                                print(f"Invalid argument error: {e}")
                                yield format_sse(f'<div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm"><strong class="font-semibold">Invalid Request:</strong> The file or request format is not supported. Please try a different file or check that the file is not corrupted.</div>')
                                return # Stop processing immediately
                            
                            # Check for 403 Forbidden (File permission/expiration)
                            if "403" in error_str or "permission to access" in error_str.lower():
                                print(f"Permission denied (expired file?): {e}")
                                yield format_sse(f'<div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm"><strong class="font-semibold">Access Denied:</strong> A file in your conversation history is no longer accessible (it may have expired). Please <button hx-post="/reset" hx-target="#chat-messages" hx-swap="innerHTML" class="underline font-bold hover:text-red-800">Reset Chat</button> to continue.</div>')
                                return # Stop processing immediately

                            # Check for 429 in string representation just in case
                            if "429" in error_str or "quota" in error_str.lower():
                                print(f"Quota exceeded (detected via string): {e}")
                                yield format_sse(f'<div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm"><strong class="font-semibold">Quota Exceeded:</strong> You have hit the API rate limit. Please wait a moment and try again.</div>')
                                return # Stop processing immediately

                            print(f"Attempt {attempt+1}/{max_retries}: Error {e}. Retrying...")
                            yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Thinking (Retrying)...</div>')
                            if attempt < max_retries - 1:
                                await asyncio.sleep(0.5)
                    
                    try:
                        # Check if we have candidates after retries
                        if not response or not response.candidates:
                            feedback = response.prompt_feedback if response else "No response"
                            print(f"No candidates returned after retries. Feedback: {feedback}")
                            yield format_sse("<div><strong>Error:</strong> No response from AI (Safety Block or API Error).</div>")
                            return

                        # Check for function call in the full response
                        if not response.candidates[0].content.parts:
                             print("Candidate returned but no parts after retries.")
                             yield format_sse("<div><strong>Error:</strong> Empty response content.</div>")
                             return

                        part = response.candidates[0].content.parts[0]
                    
                        if part.function_call:
                            print(f"Function call detected: {part.function_call.name}")
                            fn_name = part.function_call.name
                            fn_args = dict(part.function_call.args)
                            
                            # Notify user we are searching
                            yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Executing: {fn_name}...</div>')
                            
                            # Execute tool
                            if fn_name == "search_web":
                                api_response = search_web(fn_args.get("query"))
                            elif fn_name == "crawl_website":
                                api_response = crawl_website(
                                    fn_args.get("url"),
                                    fn_args.get("max_depth", 2),
                                    fn_args.get("limit", 10),
                                    fn_args.get("instructions")
                                )
                            elif fn_name == "read_source_code":
                                api_response = read_source_code(fn_args.get("file_path"))
                            elif fn_name == "write_source_code":
                                api_response = write_source_code(fn_args.get("file_path"), fn_args.get("code"))
                            elif fn_name == "import_package":
                                api_response = import_package(fn_args.get("package_name"))
                            elif fn_name == "execute_calculation":
                                api_response = execute_calculation(fn_args.get("code"))
                            elif fn_name == "generate_chart":
                                api_response = generate_chart(fn_args.get("code"))
                            elif fn_name == "generate_plotly_chart":
                                api_response = generate_plotly_chart(fn_args.get("code"))
                            elif fn_name == "generate_image":
                                api_response = generate_image(fn_args.get("description"))
                            elif fn_name == "get_current_datetime":
                                timezone_arg = fn_args.get("timezone")
                                api_response = get_current_datetime(timezone_arg) if timezone_arg else get_current_datetime()
                            elif fn_name == "get_user_timezone":
                                api_response = get_user_timezone()
                            elif fn_name == "get_coordinates":
                                api_response = get_coordinates(fn_args.get("location"))
                            elif fn_name == "get_weather":
                                api_response = get_weather(fn_args.get("location"))
                            elif fn_name == "get_forecast_weather":
                                api_response = get_forecast_weather(fn_args.get("location"))
                            elif fn_name == "get_precipitation_timing":
                                api_response = get_precipitation_timing(fn_args.get("location"))
                            elif fn_name == "wolfram_alpha_query":
                                api_response = wolfram_alpha_query(fn_args.get("query"))
                            else:
                                api_response = f"Error: Unknown tool '{fn_name}'"
                                
                            # Update history with function call and response
                            
                            # 1. Add the assistant's function call
                            messages_payload.append({
                                "role": "model",
                                "parts": [part]
                            })
                            
                            # SAVE FUNCTION CALL TO DB
                            fc_json = json.dumps({
                                "function_call": {
                                    "name": fn_name,
                                    "args": fn_args
                                }
                            })
                            save_message(session_id, "model", fc_json)
                        
                            # Display code block for code execution tools
                            if fn_name in ["execute_calculation", "generate_chart", "generate_plotly_chart"]:
                                code = fn_args.get("code", "")
                                if code:
                                    # Use OOB swap to insert the code block before the streaming response
                                    code_html = f"""
                                    <div id="code-block-{stream_id}" hx-swap-oob="beforebegin:#{stream_id}" class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                                        <details class="group">
                                            <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                                                <span class="transition group-open:rotate-90">▶</span>
                                                <span>🔧 Code executed via <code class="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">{fn_name}</code></span>
                                            </summary>
                                            <div class="mt-3">
                                                <pre class="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm"><code class="language-python">{code}</code></pre>
                                            </div>
                                        </details>
                                    </div>
                                    """
                                    yield format_sse(code_html)
                            
                            # Display file content for read_source_code and write_source_code tools
                            if fn_name in ["read_source_code", "write_source_code"]:
                                file_path = fn_args.get("file_path", "unknown_file")
                                
                                # Determine content to show
                                content_to_show = ""
                                action_label = ""
                                
                                if fn_name == "read_source_code":
                                    content_to_show = api_response
                                    action_label = "Content of"
                                else: # write_source_code
                                    content_to_show = fn_args.get("code", "")
                                    action_label = "Wrote to"
                                
                                # Only show if we have content and no error (for read)
                                if content_to_show and not (fn_name == "read_source_code" and content_to_show.startswith("Error")):
                                    # Use OOB swap to insert the code block before the streaming response
                                    code_html = f"""
                                    <div id="file-content-{stream_id}" hx-swap-oob="beforebegin:#{stream_id}" class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                                        <details class="group" open>
                                            <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                                                <span class="transition group-open:rotate-90">▶</span>
                                                <span>📄 {action_label} <code class="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">{os.path.basename(file_path)}</code></span>
                                            </summary>
                                            <div class="mt-3">
                                                <pre class="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm"><code class="language-python">{content_to_show}</code></pre>
                                            </div>
                                        </details>
                                    </div>
                                    """
                                    yield format_sse(code_html)
                            
                            # Display sources block for search_web and crawl_website tools
                            if fn_name in ["search_web", "crawl_website"]:
                                # Store original response BEFORE we modify api_response
                                original_api_response = api_response
                                try:
                                    # Parse the JSON response to extract sources
                                    search_data = json.loads(api_response)
                                    sources = search_data.get("sources", [])
                                    if sources:
                                        source_label = "🔍 Sources used for web search" if fn_name == "search_web" else "🕸️ Pages crawled from website"
                                        sources_html = f"""
                                        <div id="sources-block-{stream_id}" hx-swap-oob="beforebegin:#{stream_id}" class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                                            <details class="group">
                                                <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                                                    <span class="transition group-open:rotate-90">▶</span>
                                                    <span>{source_label}</span>
                                                </summary>
                                                <div class="mt-3 space-y-2">
                                        """
                                        
                                        for idx, source in enumerate(sources, 1):
                                            url = source.get('url', '')
                                            title = source.get('title', url)
                                            sources_html += f"""
                                            <div class="flex items-start gap-2 text-sm">
                                                <span class="text-gray-500 dark:text-gray-400 font-mono">{idx}.</span>
                                                <a href="{url}" target="_blank" rel="noopener noreferrer" 
                                                   class="text-blue-600 dark:text-blue-400 hover:underline flex-1 break-all">
                                                    {title}
                                                </a>
                                            </div>
                                            """
                                        
                                        sources_html += """
                                                </div>
                                            </details>
                                        </div>
                                        """
                                        yield format_sse(sources_html)
                                    
                                    # Extract just the context for the model
                                    api_response = search_data.get("context", api_response)
                                except json.JSONDecodeError:
                                    # If not JSON, use as-is (error message)
                                    pass
                            
                            # 2. Add the function response (send context-only to model for search_web)
                            messages_payload.append({
                                "role": "function",
                                "parts": [{
                                    "function_response": {
                                        "name": fn_name,
                                        "response": {"result": api_response}
                                    }
                                }]
                            })
                            
                            # SAVE FUNCTION RESPONSE TO DB
                            # For search_web and crawl_website, save the ORIGINAL full response (with sources) to the DB
                            # so that sources can be re-rendered on page load
                            db_response = original_api_response if fn_name in ["search_web", "crawl_website"] else api_response
                            fr_json = json.dumps({
                                "function_response": {
                                    "name": fn_name,
                                    "response": {"result": db_response}
                                }
                            })
                            save_message(session_id, "function", fr_json)
                        
                            # CHART DISPLAY: If a chart was generated (matplotlib or plotly), display it directly
                            if fn_name in ["generate_chart", "generate_plotly_chart"]:
                                try:
                                    # Check if response is valid JSON
                                    try:
                                        resp_data = json.loads(api_response)
                                    except json.JSONDecodeError:
                                        print(f"Warning: generate_chart returned non-JSON response: {api_response}")
                                        resp_data = None
                                        
                                    if isinstance(resp_data, dict):
                                        # Initialize response text components
                                        history_parts = []
                                        
                                        # 1. Handle PNG Image (Preview)
                                        if "image_path" in resp_data:
                                            chart_path = resp_data["image_path"]
                                            print(f"Chart generated at {chart_path}.")
                                            
                                            # Display chart to user immediately
                                            yield format_sse(f'<div class="mb-2"><img src="{chart_path}" alt="Generated Chart" class="max-w-full h-auto rounded-lg shadow-md"/></div>')
                                            
                                            history_parts.append(f"![Generated Chart]({chart_path})")
                                        
                                        # 2. Handle HTML Link (Interactive)
                                        if "html_path" in resp_data:
                                            html_path = resp_data["html_path"]
                                            # Use the message from response if available (contains the styled link), or create one
                                            message = resp_data.get("message", f'<a href="{html_path}" onclick="event.stopPropagation(); event.preventDefault(); window.open(this.href, \'_blank\'); return false;" class="text-blue-500 hover:underline">🔗 Open interactive chart</a>')
                                            print(f"Interactive chart generated at {html_path}.")
                                            
                                            # Display link to user
                                            yield format_sse(f'<div class="mb-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">{message}</div>')
                                            
                                            # Add to history (using special syntax we added preprocessing for, or just a link)
                                            # We'll use the link format since we have the image above
                                            history_parts.append(f"\nInteractive Version: [{html_path}]({html_path})")
                                        
                                        # 3. Finalize
                                        if history_parts:
                                            full_response_text = "\n".join(history_parts) + "\n\n*Chart generated successfully.*"
                                            
                                            # Notify UI
                                            yield format_sse(f'<div class="text-xs text-gray-500 mb-2 italic">Chart generated successfully.</div>')
                                            
                                            # Break - chart/link is displayed
                                            break
                                        
                                except Exception as e:
                                    print(f"Error displaying chart: {e}")
                                    traceback.print_exc()
                            
                            # IMAGE DISPLAY: If an image was generated, display it directly
                            if fn_name == "generate_image":
                                try:
                                    print(f"Image generation response: {api_response}")
                                    # The api_response contains markdown: ![description](url)
                                    # Extract the URL and display as an img tag
                                    import re
                                    match = re.search(r'!\[([^\]]*)\]\(([^\)]+)\)', api_response)
                                    if match:
                                        description = match.group(1)
                                        image_url = match.group(2)
                                        print(f"Extracted image URL: {image_url}")
                                        
                                        # Display image directly
                                        yield format_sse(f'<div class="mb-2"><img src="{image_url}" alt="{description}" class="max-w-full h-auto rounded-lg shadow-md"/></div>')
                                        
                                        # Set response text for history (keep as markdown so it persists)
                                        full_response_text = api_response
                                        
                                        # Break out of loop
                                        print("Breaking out of loop after image display")
                                        break
                                    else:
                                        print("Failed to parse markdown image")
                                        # Fallback if markdown parsing fails
                                        yield format_sse(f'<div class="mb-2">{api_response}</div>')
                                        full_response_text = api_response
                                        break
                                    
                                except Exception as e:
                                    print(f"Error displaying image: {e}")
                                    traceback.print_exc()
                            
                            # CONTINUE LOOP to let model use the result
                            continue
                        
                        else:
                            # No function call, just text
                            if part.text:
                                full_response_text = part.text
                                yield format_sse(full_response_text)
                                break # Exit loop, we have the final answer
                    
                    except Exception as e:
                        print(f"Error processing response: {e}")
                        yield format_sse(f"Error: {str(e)}")
                        break
                    
                    # If we broke from the parts loop and have a response, exit turn loop
                    if full_response_text:
                        break

            # 3. Save to Session History
            # Ensure we ALWAYS save a model response to prevent history corruption (dangling user messages)
            if not full_response_text:
                print("Warning: Full response text is empty. Saving placeholder.")
                full_response_text = "(No response provided by AI)"
                # Also yield it to the UI so the user sees something
                yield format_sse("<div><em>(The AI processed the request but returned no text.)</em></div>")

            save_message(session_id, "model", full_response_text)
            
            # 5. Check for auto-continue (retry phrases)
            # Get last 200 chars and check for retry phrases
            response_end = full_response_text[-200:].lower() if len(full_response_text) > 200 else full_response_text.lower()
            
            for phrase in RETRY_PHRASES:
                if phrase in response_end:
                    print(f"Auto-continue triggered: detected '{phrase}' in response")
                    auto_continue_count += 1
                    if auto_continue_count <= max_auto_continues:
                        should_auto_continue = True
                        # Add model response to payload
                        messages_payload.append({
                            "role": "model",
                            "parts": [{"text": full_response_text}]
                        })
                        # Add synthetic "continue" user message
                        continue_msg = "continue"
                        messages_payload.append({
                            "role": "user",
                            "parts": [{"text": continue_msg}]
                        })
                        # For now commented out : Save the continue message to history
                        # save_message(session_id, "user", continue_msg)
                        
                        # Notify UI
                        yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Bear with me, I am fixing encountered issues ({auto_continue_count}/{max_auto_continues})...</div>')
                    break  # Only trigger once per response
            
            # If not auto-continuing, finalize
            if not should_auto_continue:
                # 4. Finalize UI (OOB Swap)
                final_div = render_bot_message(full_response_text, stream_id=stream_id, final=True)
                
                yield format_sse(final_div)
                yield format_sse("", event="close")

        except Exception as e:
            print(f"CRITICAL ERROR in stream_response: {e}")
            traceback.print_exc()
            yield format_sse(error_div)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/reset")
@retry_on_db_lock()
def reset_chat(session_id: str = Cookie(None)):
    image_paths = []
    if session_id:
        # Separate connection scope - commit and close immediately
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            # Get all image paths for this session before deleting
            cursor = conn.execute(
                "SELECT image_path FROM messages WHERE session_id = ? AND image_path IS NOT NULL", 
                (session_id,)
            )
            image_paths = [row["image_path"] for row in cursor.fetchall()]
            
            # Delete messages from database
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()  # Explicit commit to release lock immediately
        # Connection is now closed, lock released
        
        # Delete associated files AFTER database is unlocked
        for image_path in image_paths:
            # image_path is like "/static/uploads/filename.ext"
            # Convert to filesystem path
            file_path = image_path.lstrip("/")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Clean up generated files for this session
        try:
            # Open NEW connection for generated files (previous conn is closed)
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as gen_conn:
                gen_conn.row_factory = sqlite3.Row
                # Get generated files for this session
                cursor = gen_conn.execute(
                    "SELECT file_path FROM generated_files WHERE session_id = ?", 
                    (session_id,)
                )
                generated_files = [row["file_path"] for row in cursor.fetchall()]
            
            for file_path in generated_files:
                # file_path is like "/static/generated/..."
                # Convert to filesystem path
                fs_path = file_path.lstrip("/")
                if os.path.exists(fs_path):
                    try:
                        os.remove(fs_path)
                        print(f"Deleted generated file: {fs_path}")
                    except Exception as e:
                        print(f"Error deleting generated file {fs_path}: {e}")
            
            # Delete records from DB
            gen_conn.execute("DELETE FROM generated_files WHERE session_id = ?", (session_id,))
            gen_conn.commit()
            
        except Exception as e:
            print(f"Error cleaning generated files: {e}")
    
    return ""

@app.post("/clear-file-context")
def clear_file_context(session_id: str = Cookie(None)):
    """Remove all file references from the conversation history."""
    if session_id:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            # Clear gemini_uri for all messages in this session
            conn.execute(
                "UPDATE messages SET gemini_uri = NULL WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
        
        return HTMLResponse(
            '<div class="text-xs text-green-600 mb-2 flex items-center gap-1">'
            '<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>'
            '✅ File context cleared. Tools re-enabled!'
            '</div>'
        )
    return HTMLResponse("")


@app.get("/settings/system_instruction", response_class=HTMLResponse)
async def get_system_instruction_ui(selected_model: str = Cookie("gemini-2.5-flash")):
    instruction = get_system_instruction()
    return f"""
    <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in" id="settings-modal">
        <div class="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden transform transition-all border border-gray-100 dark:border-gray-700">
            <div class="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-900/50">
                <h3 class="text-lg font-semibold text-gray-800 dark:text-gray-100">System Settings</h3>
                <button onclick="document.getElementById('settings-modal').remove()" class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            <div class="p-6 space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">System Instruction (Persona)</label>
                    <textarea name="instruction" 
                              class="w-full h-32 p-3 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 dark:focus:border-blue-400 transition-all text-sm resize-none text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500"
                              placeholder="e.g., You are a helpful assistant...">{instruction}</textarea>
                    <p class="text-xs text-gray-500 dark:text-gray-400 mt-2">This instruction will be applied to all new messages.</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Model Selection</label>
                    <select name="model" 
                            class="w-full p-3 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 dark:focus:border-blue-400 transition-all text-sm text-gray-800 dark:text-gray-200">
                        <option value="gemini-2.5-flash" {'selected' if selected_model == 'gemini-2.5-flash' else ''}>Gemini 2.5 Flash (Fastest)</option>
                        <option value="gemini-2.5-flash-lite" {'selected' if selected_model == 'gemini-2.5-flash-lite' else ''}>Gemini 2.5 Flash Lite</option>
                        <option value="gemini-2.5-flash-tts" {'selected' if selected_model == 'gemini-2.5-flash-tts' else ''}>Gemini 2.5 Flash TTS</option>
                        <option value="gemini-2.5-pro" {'selected' if selected_model == 'gemini-2.5-pro' else ''}>Gemini 2.5 Pro (Balanced)</option>
                        <option value="gemini-3-pro" {'selected' if selected_model == 'gemini-3-pro' else ''}>Gemini 3 Pro (Most Capable)</option>
                    </select>
                </div>

                <div class="border-t border-gray-100 dark:border-gray-700 pt-4">
                    <details class="group">
                        <summary class="list-none flex justify-between items-center cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                            <span>Archived Chats</span>
                            <span class="transition group-open:rotate-180">
                                <svg fill="none" height="24" shape-rendering="geometricPrecision" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" viewBox="0 0 24 24" width="24"><path d="M6 9l6 6 6-6"></path></svg>
                            </span>
                        </summary>
                        <div class="text-gray-500 dark:text-gray-400 mt-3 group-open:animate-fadeIn">
                            <div id="archived-list" hx-get="/archived-chats" hx-trigger="load" class="max-h-40 overflow-y-auto space-y-1">
                                <div class="flex justify-center p-2">
                                    <svg class="animate-spin h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                </div>
                            </div>
                        </div>
                    </details>
                </div>
            </div>
            <div class="p-6 border-t border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex justify-end gap-3">
                <button onclick="document.getElementById('settings-modal').remove()" 
                        class="px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors">
                    Cancel
                </button>
                <button hx-post="/settings/system_instruction" 
                        hx-include="[name='instruction'], [name='model']"
                        hx-target="#settings-modal"
                        hx-swap="outerHTML"
                        class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors">
                    Save Changes
                </button>
            </div>
        </div>
    </div>
    """

@app.post("/settings/system_instruction", response_class=HTMLResponse)
async def post_system_instruction(instruction: str = Form(...), model: str = Form(...)):
    save_system_instruction(instruction)
    response = Response(content="""
    <div id="settings-modal" class="fixed inset-0 z-50 flex items-end justify-center p-4 pointer-events-none">
        <div class="bg-green-50 text-green-700 border border-green-200 px-6 py-4 rounded-xl shadow-lg flex items-center gap-3 animate-fade-in-up pointer-events-auto">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
            </svg>
            <span class="font-medium">Settings saved successfully!</span>
            <script>
                setTimeout(() => {
                    const modal = document.getElementById('settings-modal');
                    if(modal) {
                        modal.style.opacity = '0';
                        setTimeout(() => modal.remove(), 300);
                    }
                }, 2000);
            </script>
        </div>
    </div>
    """)
    response.set_cookie(key="selected_model", value=model)
    return response