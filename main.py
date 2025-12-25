import os
import asyncio
import uuid
import sqlite3
import markdown

try:
    from xhtml2pdf import pisa
except ImportError:
    pisa = None
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
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted

# Custom Providers
from providers.gemini import GeminiProvider
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
import pandas as pd
# Configure Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Context variable to store client IP
client_ip_ctx = contextvars.ContextVar("client_ip", default=None)
session_id_ctx = contextvars.ContextVar("session_id", default=None)
execution_mode_ctx = contextvars.ContextVar("execution_mode", default="local")

# Retry phrases that trigger auto-continue (case-insensitive check)
RETRY_PHRASES = [
    "please wait",
    "give me a moment",
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
    "working on it",
    "one moment",
    "give me a moment",
    "let me ",
    "i'll fix",
    "i will fix",
    "i'll adjust",
    "i will adjust",
    "i'll retry",
    "i will retry",
    "i will",
    "stand by",
    "might take a few moments",
    "here we go",
    "let's",
    "I'll now",
    "I will now",
    "I'll calculate",
    "I will calculate",
    "Here is"
    "first, I need",
    "first I need",
    "Continuing..."
]

# Custom Providers
from providers.factory import get_llm_provider
from tavily import TavilyClient
import urllib.parse
import json
import uuid
import asyncio
import traceback
from contextlib import asynccontextmanager
import io
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse
from fastapi import Depends, HTTPException, status

# --- 1. CONFIGURATION ---
load_dotenv()
# Initialize Providers
llm_provider = None
# if not provided in env, use gemini 
active_provider_name = os.getenv("LLM_PROVIDER", "gemini")

if active_provider_name:
    try:
        llm_provider = get_llm_provider(active_provider_name)
    except Exception as e:
        print(f"Error initializing {active_provider_name} provider: {e}")

# Initialize Tavily Client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# Configure Tavily
tavily_client = None
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if tavily_api_key:
    tavily_client = TavilyClient(api_key=tavily_api_key)
    print(f"Tavily Search: Enabled (Key found: {tavily_api_key[:4]}...)")
else:
    print("Tavily Search: Disabled (TAVILY_API_KEY not found or empty)")

SYSTEM_INSTRUCTION_FILE = "system_instruction.txt"

def get_system_instruction(user_id: Optional[int] = None) -> str:
    # 1. Try to get from User record
    if user_id:
        try:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                cursor = conn.execute("SELECT system_instruction FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    return row[0]
        except Exception as e:
            print(f"Error fetching user instruction: {e}")

    # 2. Fallback to global file or default
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
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", os.urandom(24).hex()), https_only=False, same_site="lax")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize OAuth
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'prompt': 'select_account',  # Force account selection
    }
)

# --- 2. DATABASE & SESSION MANAGEMENT ---
DB_NAME = os.path.abspath("chat.db")
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
        conn.execute("PRAGMA foreign_keys = ON")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_sub TEXT UNIQUE,
                email TEXT,
                name TEXT,
                picture TEXT,
                intervals_athlete_id TEXT,
                intervals_api_key TEXT,
                dark_mode INTEGER DEFAULT 1,
                system_instruction TEXT,
                selected_model TEXT DEFAULT 'gemini-2.5-flash',
                coach_mode_enabled INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: Add settings columns if they don't exist
        try:
            conn.execute("ALTER TABLE users ADD COLUMN dark_mode INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN system_instruction TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN selected_model TEXT DEFAULT 'gemini-2.5-flash'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN coach_mode_enabled INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN automatic_function_calling INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                user_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_archived INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # Migration: Add is_archived column if it doesn't exist
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN is_archived INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Column likely exists

        # Migration: Add user_id column if it doesn't exist
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER REFERENCES users(id)")
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
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_code (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Workouts with CASCADE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                external_id TEXT UNIQUE,
                icu_event_id INTEGER,
                start_date_local TEXT,
                filename TEXT,
                file_contents TEXT,
                file_contents_confirmed TEXT,
                training_plan_days_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(training_plan_days_id) REFERENCES training_plan_days(id) ON DELETE CASCADE
            )
        """)
        
        # User Memories table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Migration: Ensure updated_at exists if we ever change it later
        # (Though not needed for a new table, it's good practice)
        
        # Migration: Check if CASCADE exists in workouts OR if it's pointing to a renamed table (e.g. tpd_old)
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='workouts'")
        row = cursor.fetchone()
        if row and ("ON DELETE CASCADE" not in row[0] or "tpd_old" in row[0]):
            print("Migrating workouts table for ON DELETE CASCADE and correct references...")
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("ALTER TABLE workouts RENAME TO workouts_old")
            conn.execute("""
                CREATE TABLE workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    external_id TEXT UNIQUE,
                    icu_event_id INTEGER,
                    start_date_local TEXT,
                    filename TEXT,
                    file_contents TEXT,
                    file_contents_confirmed TEXT,
                    training_plan_days_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(training_plan_days_id) REFERENCES training_plan_days(id) ON DELETE CASCADE
                )
            """)
            conn.execute("INSERT INTO workouts SELECT * FROM workouts_old")
            conn.execute("DROP TABLE workouts_old")
            conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                goal TEXT,
                methodology TEXT,
                start_date TEXT,
                duration_weeks INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Weeks with CASCADE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_plan_weeks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER,
                week_number INTEGER,
                phase TEXT,
                focus TEXT,
                total_volume_estimate TEXT,
                FOREIGN KEY(plan_id) REFERENCES training_plans(id) ON DELETE CASCADE
            )
        """)
        
        # Migration: Check if CASCADE exists in training_plan_weeks
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='training_plan_weeks'")
        row = cursor.fetchone()
        if row and "ON DELETE CASCADE" not in row[0]:
            print("Migrating training_plan_weeks table for ON DELETE CASCADE...")
            conn.execute("ALTER TABLE training_plan_weeks RENAME TO tpw_old")
            conn.execute("""
                CREATE TABLE training_plan_weeks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER,
                    week_number INTEGER,
                    phase TEXT,
                    focus TEXT,
                    total_volume_estimate TEXT,
                    FOREIGN KEY(plan_id) REFERENCES training_plans(id) ON DELETE CASCADE
                )
            """)
            conn.execute("INSERT INTO training_plan_weeks SELECT * FROM tpw_old")
            conn.execute("DROP TABLE tpw_old")

        # Days with CASCADE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_plan_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_id INTEGER,
                day_name TEXT,
                workout_type TEXT,
                details TEXT,
                minutes INTEGER,
                tss INTEGER,
                intensity_factor REAL,
                rpe INTEGER,
                planned_date TEXT,
                FOREIGN KEY(week_id) REFERENCES training_plan_weeks(id) ON DELETE CASCADE
            )
        """)
        
        # Migration: Check if CASCADE exists in training_plan_days
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='training_plan_days'")
        row = cursor.fetchone()
        if row and "ON DELETE CASCADE" not in row[0]:
            print("Migrating training_plan_days table for ON DELETE CASCADE...")
            conn.execute("ALTER TABLE training_plan_days RENAME TO tpd_old")
            conn.execute("""
                CREATE TABLE training_plan_days (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_id INTEGER,
                    day_name TEXT,
                    workout_type TEXT,
                    details TEXT,
                    minutes INTEGER,
                    tss INTEGER,
                    intensity_factor REAL,
                    rpe INTEGER,
                    planned_date TEXT,
                    FOREIGN KEY(week_id) REFERENCES training_plan_weeks(id) ON DELETE CASCADE
                )
            """)
            conn.execute("INSERT INTO training_plan_days SELECT * FROM tpd_old")
            conn.execute("DROP TABLE tpd_old")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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
            
            # For orphaned chats, we leave user_id as NULL (or assign to default if desired later)
            conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (session_id, title))
        
        print("Database initialized with WAL mode enabled")

# Initialize DB on startup
init_db()

def get_conversations(user_id: Optional[int] = None, limit: int = 50, include_archived: bool = False, only_archived: bool = False):
    """Get conversations, optionally filtering out archived ones."""
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        # User segregation: 
        # If user_id is provided, show their chats.
        # If user_id is None (anonymous), show chats with user_id IS NULL (orphaned/anonymous).
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        else:
             query += " AND user_id IS NULL"

        if only_archived:
            query += " AND is_archived = 1"
        elif not include_archived:
            query += " AND is_archived = 0"
            
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

def delete_chat_files(session_id: str):
    """Delete all files associated with a chat session."""
    import os
    
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        # Get generated files
        cursor = conn.execute("SELECT file_path FROM generated_files WHERE session_id = ?", (session_id,))
        generated_files = [row[0] for row in cursor.fetchall()]
        
        # Get uploaded files
        cursor= conn.execute("SELECT file_path FROM uploaded_files WHERE session_id = ?", (session_id,))
        uploaded_files = [row[0] for row in cursor.fetchall()]

        # get generated code files
        cursor = conn.execute("SELECT file_path FROM generated_code WHERE session_id = ?", (session_id,))
        generated_code_files = [row[0] for row in cursor.fetchall()]
        

        # Delete physical files
        all_files = generated_files + uploaded_files + generated_code_files
        for file_path in all_files:
            print(file_path)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                else:
                    print(f"File does not exist: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Delete database records
        conn.execute("DELETE FROM generated_files WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM uploaded_files WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM generated_code WHERE session_id = ?", (session_id,))
        conn.commit()

@retry_on_db_lock()
def get_setting(key: str, default: str = None) -> str:
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default
    except Exception as e:
        print(f"Error reading setting {key}: {e}")
        return default

def save_setting(key: str, value: str):
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit() # Commit the change
    except Exception as e:
        print(f"Error saving setting {key}: {e}")

def save_user_setting(user_id: int, key: str, value: any):
    """Saves a user-specific setting (dark_mode, selected_model, system_instruction, coach_mode_enabled, automatic_function_calling)."""
    if key not in ['dark_mode', 'selected_model', 'system_instruction', 'coach_mode_enabled', 'automatic_function_calling']:
        print(f"Unknown user setting: {key}")
        return
        
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute(f"UPDATE users SET {key} = ? WHERE id = ?", (value, user_id))
            conn.commit()
    except Exception as e:
        print(f"Error saving user setting {key}: {e}")

def get_user_settings(user_id: int) -> dict:
    """Retrieves all settings for a specific user."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT dark_mode, selected_model, system_instruction, coach_mode_enabled, automatic_function_calling FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                settings = dict(row)
                # Convert INTEGER to bool
                settings['coach_mode_enabled'] = bool(settings.get('coach_mode_enabled', 0))
                settings['automatic_function_calling'] = bool(settings.get('automatic_function_calling', 1))
                # If instruction is empty, fallback to global
                if not settings.get('system_instruction'):
                    settings['system_instruction'] = get_system_instruction(None)
                return settings
    except Exception as e:
        print(f"Error fetching user settings: {e}")
    
    # Defaults
    return {
        "dark_mode": 1,
        "selected_model": "gemini-2.5-flash",
        "system_instruction": get_system_instruction(None),
        "coach_mode_enabled": False
    }

def add_user_memory(content: str):
    """
    Saves a persistent memory about the user. 
    Use this when the user shares personal information, preferences, or explicitly asks you to remember something.
    
    Args:
        content (str): The information to remember.
    """
    import sqlite3
    session_id = session_id_ctx.get()
    user_id = None
    
    try:
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row[0]
        
        if not user_id:
            return "Error: Could not determine user ID. Memory not saved."

        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("INSERT INTO user_memories (user_id, content) VALUES (?, ?)", (user_id, content))
            conn.commit()
            
        return f"Successfully saved memory: {content}"
    except Exception as e:
        print(f"Error adding user memory: {e}")
        return f"Error: {e}"

def get_user_memories(user_id: int) -> List[Dict]:
    """Retrieves all memories for a specific user."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT id, content, created_at FROM user_memories WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching user memories: {e}")
        return []

def delete_user_memory(memory_id: int, user_id: int):
    """Deletes a specific memory record."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("DELETE FROM user_memories WHERE id = ? AND user_id = ?", (memory_id, user_id))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error deleting user memory: {e}")
        return False

def update_user_memory(memory_id: int, user_id: int, content: str):
    """Updates the content of a specific memory record."""
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE user_memories SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?", (content, memory_id, user_id))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error updating user memory: {e}")
        return False

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
                tool_calls = data.get("function_call")
                tool_responses = data.get("function_response")
                if tool_calls or tool_responses:
                    if tool_calls:
                        if not isinstance(tool_calls, list):
                            tool_calls = [tool_calls]
                        for tc in tool_calls:
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc['name'],
                                        args=tc['args']
                                    )
                                )
                            )
                    
                    if tool_responses:
                        if not isinstance(tool_responses, list):
                            tool_responses = [tool_responses]
                        for tr in tool_responses:
                            # Map old response format to new SDK dict requirement
                            resp_val = tr.get('response')
                            # If it's already a dict, use it, otherwise wrap it
                            if not isinstance(resp_val, dict):
                                resp_val = {"result": resp_val}
                                
                            parts.append(
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=tr['name'],
                                        response=resp_val
                                    )
                                )
                            )
                else:
                    parts.append(types.Part(text=content))
            except (json.JSONDecodeError, KeyError, TypeError):
                parts.append(types.Part(text=content))
        else:
            parts.append(types.Part(text=content))

        # Add image if present
        image_path = row["image_path"]
        # Check if gemini_uri exists in keys (for safety during migration)
        gemini_uri = row["gemini_uri"] if "gemini_uri" in row.keys() else None
        mime_type = row["mime_type"] if "mime_type" in row.keys() else "image/jpeg" # Default fallback
        
        if gemini_uri:
            # Use types.Part with FileData
            parts.append(
                types.Part(
                    file_data=types.FileData(
                        mime_type=mime_type,
                        file_uri=gemini_uri
                    )
                )
            )
        elif image_path:
            # Fallback for providers that don't use Gemini File API (e.g. OpenAI)
            # We treat the local path as the URI for internal processing
            parts.append(
                types.Part(
                    file_data=types.FileData(
                        mime_type=mime_type,
                        file_uri=image_path 
                    )
                )
            )

        history.append({"role": role, "parts": parts})
    return history

@retry_on_db_lock()
def save_message(session_id: str, role: str, content: str, image_path: Optional[str] = None, mime_type: Optional[str] = None, gemini_uri: Optional[str] = None, user_id: Optional[int] = None):
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        # Ensure conversation exists
        cursor = conn.execute("SELECT id, user_id FROM conversations WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            title = "New Chat"
            if role == "user":
                title = content[:30] + "..." if len(content) > 30 else content
            conn.execute("INSERT INTO conversations (id, title, user_id) VALUES (?, ?, ?)", (session_id, title, user_id))
        elif role == "user":
             # Update updated_at
             conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
             
             # Check if we need to associate an anonymous chat with the logged-in user (optional auto-claim?) 
             # For now, we respect the existing link.
             
             # Update title if it's still generic
             cursor = conn.execute("SELECT title FROM conversations WHERE id = ?", (session_id,))
             current_title = cursor.fetchone()[0]
             if current_title == "New Chat":
                 new_title = content[:30] + "..." if len(content) > 30 else content
                 conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, session_id))
        if image_path:
            conn.execute("INSERT INTO uploaded_files (session_id, file_path) VALUES (?, ?)", (session_id, image_path[1:]))
        conn.execute(
            "INSERT INTO messages (session_id, role, content, image_path, mime_type, gemini_uri) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, image_path, mime_type, gemini_uri)
        )

# --- 3. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request, response: Response):
    """
    On first load, check for a cookie. If missing, generate one.
    Also check for user session.
    """
    # Check for logged in user
    user_info = request.session.get('user')
    current_user_id = None
    if user_info:
        # Look up user_id
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
             cursor = conn.execute("SELECT id FROM users WHERE google_sub = ?", (user_info.get('sub'),))
             row = cursor.fetchone()
             if row:
                 current_user_id = row[0]
    else:
        # Mandatory Login: Redirect to login page if not authenticated
        return RedirectResponse(url='/login')

    # Load user settings
    user_settings = get_user_settings(current_user_id) if current_user_id else {
        "dark_mode": 1,
        "selected_model": "gemini-2.5-flash",
        "system_instruction": get_system_instruction(None)
    }

    # Check if user already has a session cookie
    session_id = request.cookies.get("session_id")
    
    # Ownership Check: Ensure the cookie session belongs to the logged-in user
    if session_id and current_user_id:
        if not verify_chat_ownership(session_id, current_user_id):
            # Session exists but belongs to someone else (or no one). Start fresh.
            session_id = None

    if not session_id:
        session_id = str(uuid.uuid4())

    # Load history to render
    chat_history_html = ""
    
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT role, content, image_path, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC", 
            (session_id,)
        )
        rows = cursor.fetchall()
        
        for row in rows:
            role = row["role"]
            content = row["content"]
            image_path = row["image_path"]
            timestamp = row["timestamp"]

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
                chat_history_html += render_user_message(content, image_path, timestamp)
            else:
                chat_history_html += render_bot_message(content)

    # Fetch previous conversations for the Current User (or anonymous)
    conversations = get_conversations(user_id=current_user_id)

    # Get current chat title
    current_chat_title = "New Chat"
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        cursor = conn.execute("SELECT title FROM conversations WHERE id = ?", (session_id,))
        result = cursor.fetchone()
        if result:
            current_chat_title = result[0]

    response = templates.TemplateResponse("index.html", {
        "request": request, 
        "chat_history": chat_history_html,
        "conversations": conversations,
        "current_session_id": session_id,
        "current_chat_title": current_chat_title,
        "user": user_info, # Pass user info to template
        "user_settings": user_settings
    })
    
    # Set cookie if it was missing
    if not request.cookies.get("session_id"):
        response.set_cookie(key="session_id", value=session_id, max_age=31536000) # 1 year
        
    return response

# --- Authentication & Authorization Dependencies ---
def get_current_user_id(request: Request):
    user_info = request.session.get('user')
    if not user_info:
        # Check if it's an HTMX request
        if request.headers.get("HX-Request"):
            response = Response()
            response.headers["HX-Redirect"] = "/login"
            return response
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    
    # Resolve user_id
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
         cursor = conn.execute("SELECT id FROM users WHERE google_sub = ?", (user_info.get('sub'),))
         row = cursor.fetchone()
         if row:
             return row[0]
    return None

def verify_chat_ownership(session_id: str, user_id: int):
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return False # Chat doesn't exist (or is new/ unsaved)
        if row[0] != user_id:
            return False
    return True

@app.post("/new-chat")
async def new_chat(request: Request, user_id: int = Depends(get_current_user_id)):
    """Creates a new chat session."""
    # If user_id is a redirect response, return it
    if isinstance(user_id, Response): return user_id
    
    new_session_id = str(uuid.uuid4())
    response = Response()
    response.set_cookie(key="session_id", value=new_session_id, max_age=31536000)
    response.headers["HX-Redirect"] = "/" # Redirect to home to reload sidebar and empty chat
    return response

@app.get("/load-chat/{session_id}")
async def load_chat(session_id: str, request: Request, user_id: int = Depends(get_current_user_id)):
    """Loads a specific chat session."""
    if isinstance(user_id, Response): return user_id

    # Enforce Ownership
    if not verify_chat_ownership(session_id, user_id):
         # If not owned, redirect to home (which will likely start a new chat or show error)
         response = Response()
         response.headers["HX-Redirect"] = "/"
         return response

    response = Response()
    response.set_cookie(key="session_id", value=session_id, max_age=31536000)
    response.headers["HX-Redirect"] = "/" # Redirect to home to reload everything
    return response

# --- 4. Helper Functions for Rendering ---

@app.get("/login")
async def login(request: Request):
    """Redirects to Google for authentication."""
    redirect_uri = request.url_for('auth')
    # Force http if needed (sometimes starlette detects https behind proxy)
    # redirect_uri = str(redirect_uri).replace("https://", "http://") 
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    """Callback for Google OAuth."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        # Handle error (e.g. invalid grant)
        return HTMLResponse(f"Auth Error: {e}<br><a href='/'>Go Home</a>")
        
    user_info = token.get('userinfo')
    if user_info:
        request.session['user'] = dict(user_info)
        
        # Sync user to DB
        sub = user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')
        
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            cursor = conn.execute("SELECT id FROM users WHERE google_sub = ?", (sub,))
            row = cursor.fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO users (google_sub, email, name, picture) VALUES (?, ?, ?, ?)",
                    (sub, email, name, picture)
                )
                print(f"Created new user: {email}")
            else:
                # Update info
                conn.execute(
                    "UPDATE users SET email = ?, name = ?, picture = ? WHERE google_sub = ?",
                    (email, name, picture, sub)
                )
    
    return RedirectResponse(url='/')

@app.get("/logout")
async def logout(request: Request):
    """Logs out the user."""
    request.session.pop('user', None)
    return RedirectResponse(url='/')

def render_sidebar_html(current_session_id: str, user_id: Optional[int] = None) -> str:
    """Renders the sidebar chat list HTML."""
    conversations = get_conversations(user_id=user_id)
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

def render_archived_list_html(user_id: Optional[int] = None) -> str:
    """Renders the archived chats list HTML."""
    archived = get_conversations(limit=100, only_archived=True, user_id=user_id)
    
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
async def rename_chat(session_id: str, new_title: str = Form(...), user_id: int = Depends(get_current_user_id)):
    """Rename a chat conversation."""
    if isinstance(user_id, Response): return user_id
    if not verify_chat_ownership(session_id, user_id):
        return HTMLResponse("<div>Error: Unauthorized</div>", status_code=403)

    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, session_id))
            conn.commit()
        return HTMLResponse("<div>Chat renamed successfully</div>")
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.post("/archive-chat/{session_id}")
async def archive_chat(session_id: str, request: Request, user_id: int = Depends(get_current_user_id)):
    """Archive a chat conversation."""
    if isinstance(user_id, Response): return user_id
    if not verify_chat_ownership(session_id, user_id):
        return HTMLResponse("<div>Error: Unauthorized</div>", status_code=403)

    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET is_archived = 1 WHERE id = ?", (session_id,))
            conn.commit()
        
        current_session_id = request.cookies.get("session_id")

        # If the archived chat is the current one, we need to reset the UI
        if session_id == current_session_id:
            # Generate new session like /new-chat
            new_session_id = str(uuid.uuid4())
            response = Response()
            response.set_cookie(key="session_id", value=new_session_id, max_age=31536000)
            response.headers["HX-Redirect"] = "/" # Full page reload to clean state
            return response

        # Resolve user_id for sidebar
        user_info = request.session.get('user')
        current_user_id = None
        if user_info:
             with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                 cursor = conn.execute("SELECT id FROM users WHERE google_sub = ?", (user_info.get('sub'),))
                 row = cursor.fetchone()
                 if row: current_user_id = row[0]

        sidebar_html = render_sidebar_html(current_session_id, user_id=current_user_id)
        archived_html = render_archived_list_html(user_id=current_user_id)
        
        return HTMLResponse(
            f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>'
            f'<div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>'
        )
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.post("/unarchive-chat/{session_id}")
async def unarchive_chat(session_id: str, request: Request, user_id: int = Depends(get_current_user_id)):
    """Unarchive a chat conversation."""
    if isinstance(user_id, Response): return user_id
    if not verify_chat_ownership(session_id, user_id):
        return HTMLResponse("<div>Error: Unauthorized</div>", status_code=403)

    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET is_archived = 0 WHERE id = ?", (session_id,))
            conn.commit()
            
        current_session_id = request.cookies.get("session_id")
        
        # Resolve user_id for sidebar
        user_info = request.session.get('user')
        current_user_id = None
        if user_info:
             with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                 cursor = conn.execute("SELECT id FROM users WHERE google_sub = ?", (user_info.get('sub'),))
                 row = cursor.fetchone()
                 if row: current_user_id = row[0]
                 
        sidebar_html = render_sidebar_html(current_session_id, user_id=current_user_id)
        archived_html = render_archived_list_html(user_id=current_user_id)
        
        return HTMLResponse(
            f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>'
            f'<div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>'
        )
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.delete("/delete-chat/{session_id}")
async def delete_chat_route(session_id: str, request: Request, user_id: int = Depends(get_current_user_id)):
    """Permanently delete a chat conversation and all associated files."""
    if isinstance(user_id, Response): return user_id
    if not verify_chat_ownership(session_id, user_id):
        return HTMLResponse("<div>Error: Unauthorized</div>", status_code=403)

    try:
        delete_chat_files(session_id)
        
        current_session_id = request.cookies.get("session_id")
        
        # If the deleted chat is the current one, we need to reset the UI
        if session_id == current_session_id:
            # Generate new session like /new-chat
            new_session_id = str(uuid.uuid4())
            response = Response()
            response.set_cookie(key="session_id", value=new_session_id, max_age=31536000)
            response.headers["HX-Redirect"] = "/" # Full page reload to clean state
            return response
            
        # Otherwise just update components
        # Resolve user_id for sidebar
        user_info = request.session.get('user')
        current_user_id = None
        if user_info:
             with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                 cursor = conn.execute("SELECT id FROM users WHERE google_sub = ?", (user_info.get('sub'),))
                 row = cursor.fetchone()
                 if row: current_user_id = row[0]

        sidebar_html = render_sidebar_html(current_session_id, user_id=current_user_id)
        archived_html = render_archived_list_html(user_id=current_user_id)
        
        return HTMLResponse(
            f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>'
            f'<div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>'
        )
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.get("/archived-chats")
async def get_archived_chats(user_id: int = Depends(get_current_user_id)):
    """Get list of archived chats for settings modal."""
    # Note: verify_chat_ownership isn't needed here as we aren't accessing a specific chat,
    # just listing the user's own archived chats.
    if isinstance(user_id, Response): return user_id
    
    try:
        html = render_archived_list_html(user_id=user_id)
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<div>Error: {str(e)}</div>", status_code=500)

@app.post("/save-user-settings")
async def save_user_settings_route(
    selected_model: str = Form(...),
    system_instruction: str = Form(...),
    user_id: int = Depends(get_current_user_id)
):
    if isinstance(user_id, Response): return user_id
    
    save_user_setting(user_id, 'selected_model', selected_model)
    save_user_setting(user_id, 'system_instruction', system_instruction)
    
    return HTMLResponse("<div>Settings saved successfully</div>")

@app.post("/toggle-dark-mode")
async def toggle_dark_mode_route(
    dark_mode: int = Form(...),
    user_id: int = Depends(get_current_user_id)
):
    if isinstance(user_id, Response): return user_id
    
    save_user_setting(user_id, 'dark_mode', dark_mode)
    return Response(status_code=200)

# Helper to render messages (DRY)
def render_user_message(content: str, image_path: Optional[str] = None, timestamp: Optional[str] = None) -> str:
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
            
    timestamp_html = ""
    if timestamp:
        try:
            # Format timestamp nicely (assuming SQLite format YYYY-MM-DD HH:MM:SS)
            from datetime import datetime
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            formatted_ts = dt.strftime("%b %d, %H:%M")
            timestamp_html = f'<div class="text-[10px] text-gray-400 dark:text-gray-500 mb-1 px-1 text-right w-full">{formatted_ts}</div>'
        except:
             timestamp_html = f'<div class="text-[10px] text-gray-400 dark:text-gray-500 mb-1 px-1 text-right w-full">{timestamp}</div>'

    return f"""
    <div class="flex flex-col items-end mb-4 animate-fade-in group">
        {timestamp_html}
        <div class="user-msg-bubble relative bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-sm max-w-[80%] shadow-sm transition-all duration-200 group-hover:shadow-md">
            {image_html}
            <p class="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
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
    import uuid
    content = re.sub(
        r'!\[([^\]]*)\]\(([^)]+\.html)\)',
        r'<a href="\2" onclick="event.stopPropagation(); event.preventDefault(); window.open(this.href, \'_blank\'); return false;" class="text-blue-500 hover:underline">🔗 Open interactive chart</a>',
        content
    )
    
    # 1. Protect Code Blocks first (to prevent math-masking inside code)
    # This keeps both fenced code and inline code pristine.
    code_placeholders = {}
    code_pattern = r'(```.*?```|`.*?`)'
    def mask_code(match):
        placeholder = f"MARKDOWN_CODE_PROTECTED_{uuid.uuid4().hex}__"
        code_placeholders[placeholder] = match.group(0)
        return placeholder
    
    content_no_code = re.sub(code_pattern, mask_code, content, flags=re.DOTALL)
    
    # 2. Protect Math Blocks in the code-free content
    math_placeholders = {}
    # Priority: Block math patterns first ($$...$$, \[...\]), then inline ($, \(\))
    math_pattern = r'(\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$|\\\(.*?\\\))'
    def mask_math(match):
        placeholder = f"LATEX_MATH_PROTECTED_{uuid.uuid4().hex}__"
        math_placeholders[placeholder] = match.group(0)
        return placeholder
    
    protected_content = re.sub(math_pattern, mask_math, content_no_code, flags=re.DOTALL)
    
    # 3. Restore Code Blocks (BEFORE Markdown conversion so they get processed)
    for placeholder, original in code_placeholders.items():
        protected_content = protected_content.replace(placeholder, original)

    # 4. Run Markdown on content (now including code blocks)
    html_content = markdown.markdown(protected_content, extensions=['fenced_code', 'tables'])
    
    # 5. Restore Math Blocks
    for placeholder, original in math_placeholders.items():
        html_content = html_content.replace(placeholder, original)
    
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
    Useless when running code in sandbox mode, but you would not be aware of that, but good to know.
    Args:
        package_name (str): The name of the package on PyPI (e.g., "PyYAML").
        install (bool): Whether to install the package if it is not installed. Default is True.

    Returns:
        module: The imported module object (or errormessage if failed to install)
    """
    execution_mode = execution_mode_ctx.get()
    if execution_mode == "e2b":
        return {"response": "Failure, running in sandbox code execution mode, cannot import packages with this tool, use execute_calculation, generate_chart or generate_plotly_chart instead."}
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

def write_source_code(session_id: str, file_path: str, code: str):
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
            file_path = 'static/generated_code/' + file_path
            with open(file_path, 'w') as f:
                print(f"Writing file {file_path}")
                f.write(code)
                try:
                    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                        conn.execute("INSERT INTO generated_code (session_id, file_path) VALUES (?, ?)", (session_id, file_path))
                except Exception as e:
                    print(f"Failed to insert file path into database: {e}")
                return f"Successfully wrote file {file_path}, with url : `/{file_path}`"
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
            file_path = 'static/generated_code/' + file_path
            with open(file_path, 'r') as f:
                print(f"Reading file {file_path}")
                code = f.read()
                return code
        else:
            return f"Source code file name not allowed: {file_path}"   
    except Exception as e:
        return f"Error reading file {file_path}: {e}"


def read_uploaded_file(file_name: str):
    """
    Reads the contents of a previously uploaded file from the static/uploads directory.
    Use this tool when you need to read a file that was uploaded by the user.
    Args:
        file_name (str): The name of the file to read (e.g., "data.csv"). Do not include the path.

    Returns:
        str: The contents of the file.
    """
    try:
        # Sanitize input to prevent directory traversal
        file_name = os.path.basename(file_name)
        file_path = os.path.join("static/uploads", file_name)
        
        if not os.path.exists(file_path):
            return f"Error: File '{file_name}' not found in uploads."
            
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"Reading uploaded file {file_path}")
            return f.read()
    except Exception as e:
        return f"Error reading uploaded file {file_name}: {e}"


def search_web(query: str,numberofwebsearchresults: int=4):
    """
    Searches the web for the given query using Tavily to get up-to-date information.
    Use this tool when the user asks about current events, news, or specific information 
    that might not be in your training data.
    Args:
        query (str): The search query.
        numberofwebsearchresults (int): The number of web search results to return. Default is 4. Max is 10.
    Returns:
        str: The web search results.
    """
    numberofwebsearchresults = max(numberofwebsearchresults, 10)
    if not tavily_client:
        return "Error: Web search is not configured (missing API key)."
    
    try:
        print(f"Searching web for: {query}")
        response = tavily_client.search(query=query, search_depth="basic")
        results = response.get('results', [])[:numberofwebsearchresults]  # Limit to top 3
        
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
            context_parts.append(types.Part(text=f"Page: {page_url}\nContent: {raw_content}"))
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

def convert_md_to_pdf(markdown_text: str, filename: str = None):
    """
    Converts Markdown text to a PDF file and saves it to the static/generated directory.
    Use this tool when the user asks to create a PDF document from text or markdown content.
    Args:
        markdown_text (str): The markdown content to convert.
        filename (str): Optional filename for the PDF (ending in .pdf). If not provided, a random name is generated.
    Returns:
        str: The URL to the generated PDF file.
    """
    try:
        # Convert Markdown to HTML
        html_content = markdown.markdown(markdown_text, extensions=['extra', 'codehilite'])
        
        # Add basic styling
        full_html = f"""
        <html>
        <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
            code {{ font-family: monospace; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Generate Filename
        if not filename:
            filename = f"doc_{uuid.uuid4().hex[:8]}.pdf"
        elif not filename.endswith('.pdf'):
            filename += ".pdf"
            
        # Ensure directory exists
        output_dir = "static/generated"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        # Generate PDF
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(src=full_html, dest=pdf_file)
            
        if pisa_status.err:
            return f"Error generating PDF: {pisa_status.err}"
        
        session_id = session_id_ctx.get()
        if session_id:
            try:
                with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                    conn.execute("INSERT INTO generated_files (session_id, file_path) VALUES (?, ?)", (session_id, os.path.join(GENERATED_DIR, os.path.basename(output_path))))
            except Exception as db_e:
                print(f"Error saving generated file to DB: {db_e}")
    
        return f"PDF created successfully. url: /{output_path}"
    except Exception as e:
        return f"Error converting to PDF: {str(e)}"

def rename_chat_tool(new_name: str, session_id: str = None):
    """
    Renames a chat session.
    Args:
        new_name (str): The new title for the chat.
        session_id (str, optional): The ID of the session to rename. Defaults to the current session.
    Returns:
        str: Success message or error.
    """
    target_id = session_id or session_id_ctx.get()
    if not target_id:
        return "Error: No session ID provided or found in context."
    
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_name, target_id))
            conn.commit()
        return f"Chat renamed to '{new_name}'."
    except Exception as e:
        return f"Error renaming chat: {str(e)}"

def archive_chat_tool(session_id: str = None):
    """
    Archives a chat session.
    Args:
        session_id (str, optional): The ID of the session to archive. Defaults to the current session.
    Returns:
        str: Success message or error.
    """
    target_id = session_id or session_id_ctx.get()
    if not target_id:
        return "Error: No session ID provided or found in context."
    
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("UPDATE conversations SET is_archived = 1 WHERE id = ?", (target_id,))
            conn.commit()
        return "Chat archived."
    except Exception as e:
        return f"Error archiving chat: {str(e)}"

def delete_chat_tool(session_id: str = None):
    """
    Permanently deletes a chat session.
    Args:
        session_id (str, optional): The ID of the session to delete. Defaults to the current session.
    Returns:
        str: Success message or error.
    """
    target_id = session_id or session_id_ctx.get()
    if not target_id:
        return "Error: No session ID provided or found in context."
    
    try:
        # Delete files first
        try:
            delete_chat_files(target_id)
        except Exception as e:
            print(f"Error deleting files for chat {target_id}: {e}")

        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (target_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (target_id,))
            conn.commit()
        return "Chat deleted."
    except Exception as e:
        return f"Error deleting chat: {str(e)}"

def upload_workout_to_intervals(start_date_local: str, filename: str, workout_file_content: str, training_plan_days_id: int):
    """
    Uploads a workout file (.zwo format) to Intervals.icu and saves metadata to the local database.
    
    Args:
        start_date_local (str): The planned date of the workout in ISO format (e.g., "2024-03-30T00:00:00").
        filename (str): The name of the workout file (e.g., "Workout.zwo").
        workout_file_content (str): The text content of the .zwo workout file.
        training_plan_days_id (int): The ID of the training plan day associated with the workout.
    
    Returns:
        str: A message indicating the result of the upload and database storage.
    """
    import base64
    import json
    import requests
    from requests.auth import HTTPBasicAuth
    
    # 1. Generate unique external_id
    external_id = str(uuid.uuid4())
    user_id = None
    
    # Get user_id from session or context if possible
    # In this app, user_id is often handled in routes. 
    # For tool calls, we can try to find the user_id from the current session.
    session_id = session_id_ctx.get()
    
    try:
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                # Step 1: Get user_id from session
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row['user_id']
                # Step 2: Get credentials from user
                if user_id:
                    cursor = conn.execute("SELECT intervals_athlete_id, intervals_api_key FROM users WHERE id = ?", (user_id,))
                    u_row = cursor.fetchone()
                    if u_row:
                        athlete_id = u_row['intervals_athlete_id']
                        api_key = u_row['intervals_api_key']
        if not athlete_id or not api_key:
            return "Error: Intervals.icu credentials not found for this user. Please configure them in settings."
        
        # 2. Insert into database
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("""
                INSERT INTO workouts (user_id, external_id, start_date_local, filename, file_contents, training_plan_days_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, external_id, start_date_local, filename, workout_file_content, training_plan_days_id))
            conn.commit()
            
        # 3. Base64 encode file contents
        file_contents_base64 = base64.b64encode(workout_file_content.encode('utf-8')).decode('utf-8')
        
        # 4. Call Intervals.icu API
        url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/events?upsertOnUid=true"
        payload = {
            "category": "WORKOUT",
            "start_date_local": start_date_local,
            "filename": filename,
            "file_contents_base64": file_contents_base64,
            "external_id": external_id
        }
        
        response = requests.post(
            url, 
            json=payload, 
            auth=HTTPBasicAuth('API_KEY', api_key),
            headers={'Accept': '*/*'}
        )
        
        if response.status_code in [200, 201]:
            resp_data = response.json()
            icu_event_id = resp_data.get("id")
            
            # 5. Update DB with icu_event_id
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                conn.execute("UPDATE workouts SET icu_event_id = ? WHERE external_id = ?", (icu_event_id, external_id))
                conn.commit()
            
            # 6. Confirmation GET request
            try:
                confirm_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/events/{icu_event_id}/downloadzwo"
                confirm_response = requests.get(
                    confirm_url, 
                    auth=HTTPBasicAuth('API_KEY', api_key),
                    headers={'Accept': '*/*'}
                )
                if confirm_response.status_code == 200:
                    file_contents_confirmed = confirm_response.text
                    # 7. Update DB with confirmed contents
                    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                        conn.execute("UPDATE workouts SET file_contents_confirmed = ? WHERE external_id = ?", (file_contents_confirmed, external_id))
                        conn.commit()
                    return f"Success: Workout '{filename}' uploaded and confirmed. (ICU ID: {icu_event_id}, Internal ID: {external_id})."
                else:
                    return f"Success: Workout '{filename}' uploaded (ICU ID: {icu_event_id}), but confirmation download failed. Status: {confirm_response.status_code}"
            except Exception as confirm_e:
                print(f"Error during confirmation GET: {str(confirm_e)}")
                return f"Success: Workout '{filename}' uploaded (ICU ID: {icu_event_id}), but error during confirmation download: {str(confirm_e)}"
        else:
            return f"Error: Failed to upload to Intervals.icu. Status: {response.status_code}, Response: {response.text}"
            
    except Exception as e:
        print(f"Error in upload_workout_to_intervals: {str(e)}")
        return f"Error: {str(e)}"

def delete_workout_from_intervals(start_date: str):
    """
    Deletes a workout from Intervals.icu calendar and the local database.
    if more then one workout is found on the start_date, it will delete all of them.
    
    Args:
        start_date (str): The date of the workout(s) to delete (e.g., "2024-03-30"). 
                          It will match any workout starting on this date.
    
    Returns:
        str: A message indicating the result of the deletion.
    """
    import json
    import requests
    from requests.auth import HTTPBasicAuth
    
    session_id = session_id_ctx.get()
    user_id = None
    athlete_id = None
    api_key = None
    
    try:
        # 1. Resolve user_id and Intervals.icu credentials
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                # Step 1: Get user_id from session
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row['user_id']
                # Step 2: Get credentials from user
                if user_id:
                    cursor = conn.execute("SELECT intervals_athlete_id, intervals_api_key FROM users WHERE id = ?", (user_id,))
                    u_row = cursor.fetchone()
                    if u_row:
                        athlete_id = u_row['intervals_athlete_id']
                        api_key = u_row['intervals_api_key']

        if not athlete_id or not api_key:
            return "Error: Intervals.icu credentials not found for this user. Please configure them in settings."
        
        # 2. Find the workout in the database
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            # We use LIKE with a wildcard to match the date even if it has a time component
            query = "SELECT * FROM workouts WHERE user_id IS "
            params = []
            if user_id:
                query += "? "
                params.append(user_id)
            else:
                query += "NULL "
            
            query += "AND start_date_local LIKE ? "
            params.append(f"{start_date}%")
            
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            
        if not rows:
            return f"Error: No workout found on {start_date} in the database."
        
        # 3. Call Intervals.icu API for each found workout
        all_results = []
        
        for row in rows:
            icu_event_id = row['icu_event_id']
            filename = row['filename']
            workout_id = row['id']
            url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/events/{icu_event_id}"
            
            try:
                response = requests.delete(
                    url, 
                    auth=HTTPBasicAuth('API_KEY', api_key),
                    headers={'Accept': '*/*'}
                )
                
                if response.status_code in (200,404,422):
                    # 4. Remove from local database
                    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                        conn.execute("DELETE FROM workouts WHERE id = ?", (workout_id,))
                        conn.commit()
                    all_results.append(f"Success: Workout '{filename}' (ICU ID: {icu_event_id}) deleted or not found.")
                else:
                    all_results.append(f"Error: Workout '{filename}' (ICU ID: {icu_event_id}) failed to delete. Status: {response.status_code}, Response: {response.text}")
            except Exception as api_e:
                all_results.append(f"Error: Workout '{filename}' (ICU ID: {icu_event_id}) failed due to exception: {str(api_e)}")

        return "\n".join(all_results)
            
    except Exception as e:
        print(f"Error in delete_workout_from_intervals: {str(e)}")
        return f"Error: {str(e)}"

def get_workout_from_intervals(start_date: str):
    """
    Retrieves full workout details from Intervals.icu calendar.
    
    Args:
        start_date (str): The date of the workout to retrieve (e.g., "2024-03-30"). 
                          It will match any workout starting on this date.
    
    Returns:
        str: The full JSON payload of the workout from Intervals.icu or an error message.
    """
    import json
    import requests
    from requests.auth import HTTPBasicAuth
    
    session_id = session_id_ctx.get()
    user_id = None
    athlete_id = None
    api_key = None
    
    try:
        # 1. Resolve user_id and Intervals.icu credentials
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                # Step 1: Get user_id from session
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row['user_id']
                
                # Step 2: Get credentials from user
                if user_id:
                    cursor = conn.execute("SELECT intervals_athlete_id, intervals_api_key FROM users WHERE id = ?", (user_id,))
                    u_row = cursor.fetchone()
                    if u_row:
                        athlete_id = u_row['intervals_athlete_id']
                        api_key = u_row['intervals_api_key']

        if not athlete_id or not api_key:
            return "Error: Intervals.icu credentials not found for this user. Please configure them in settings."
        
        # 2. Find the workout in the database
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM workouts WHERE user_id IS "
            params = []
            if user_id:
                query += "? "
                params.append(user_id)
            else:
                query += "NULL "
            
            query += "AND start_date_local LIKE ? AND icu_event_id IS NOT NULL"
            params.append(f"{start_date}%")
            
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            
        if not rows:
            return f"Error: No workout found on {start_date} in the database."
        
        # 3. Call Intervals.icu API for each found workout
        all_results = []
        
        for idx, row in enumerate(rows, 1):
            icu_event_id = row['icu_event_id']
            url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/events/{icu_event_id}"
            
            try:
                response = requests.get(
                    url, 
                    auth=HTTPBasicAuth('API_KEY', api_key),
                    headers={'Accept': '*/*'}
                )
                
                if response.status_code == 200:
                    payload = response.json()
                    all_results.append(f"--- Workout {idx} (ICU ID: {icu_event_id}, Filename: {row['filename']}) ---\n" + json.dumps(payload, indent=2))
                else:
                    all_results.append(f"--- Workout {idx} (ICU ID: {icu_event_id}) ---\nError retrieving: {response.status_code} {response.text}")
            except Exception as api_e:
                all_results.append(f"--- Workout {idx} (ICU ID: {icu_event_id}) ---\nException during retrieval: {str(api_e)}")

        return "\n\n".join(all_results)
            
    except Exception as e:
        print(f"Error in get_workout_from_intervals: {str(e)}")
        return f"Error: {str(e)}"

def list_workouts_from_db():
    """
    Lists all workouts in the database for the current user.
    
    Returns:
        str: A JSON string containing the list of workouts (excluding file contents).
    """
    import json
    import sqlite3
    
    session_id = session_id_ctx.get()
    user_id = None
    
    try:
        # 1. Resolve user_id
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row['user_id']
        
        # 2. Query workouts
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            # Include file_contents_confirmed to calculate workout_in_external_calendar
            query = "SELECT id, external_id, icu_event_id, start_date_local, filename, created_at, file_contents_confirmed, training_plan_days_id FROM workouts WHERE user_id IS "
            params = []
            if user_id:
                query += "? "
                params.append(user_id)
            else:
                query += "NULL "
            
            query += "ORDER BY start_date_local DESC" # Added ORDER BY for consistent results
            
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            
        if not rows:
            return "No workouts found in the database."
        
        workouts = []
        for row in rows:
            workout = dict(row)
            # Add workout_in_external_calendar boolean
            workout['workout_in_external_calendar'] = workout['file_contents_confirmed'] is not None
            # Do NOT include the 2 file contents fields in the response.
            if 'file_contents' in workout:
                del workout['file_contents']
            if 'file_contents_confirmed' in workout:
                del workout['file_contents_confirmed']
            workouts.append(workout)
            
        return json.dumps(workouts, indent=2)
            
    except Exception as e:
        print(f"Error in list_workouts_from_db: {str(e)}")
        return f"Error: {str(e)}"

def save_training_plan(plan_json: str):
    """
    Saves a hierarchical training plan to the database.
    A user can only have one training plan at a time.
    
    Args:
        plan_json (str): The training plan as a JSON string.
    
    Example:
{
  "training_plan": {
    "name": "12-Week Polarized: Endurance & Peak Power (Weekly Detail)",
    "goal": "Build deep aerobic base while maximizing neuromuscular power output.",
    "methodology": "Polarized (80/20)",
    "duration_weeks": 12,
    "schedule": [
      {
        "week_number": 1,
        "phase": "Phase 1: Base & Neuromuscular Activation",
        "focus": "Introduction to Sprint Load",
        "total_volume_estimate": "8-9 hours",
        "days": [
          { "day": "Monday", "type": "Rest", "details": "Complete Rest", "minutes": 0, "TSS":0,"IF":0,"RPE":0 },
          { "day": "Tuesday", "type": "High Intensity", "details": "6 x 10s MAX Sprints (Torque focus). Rest 5m between.", "minutes": 60,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Wednesday", "type": "Endurance", "details": "Zone 1 Steady.", "minutes": 90,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Thursday", "type": "Endurance", "details": "Zone 1 + 3 x 1m High Cadence (110rpm).", "minutes": 60,"TSS":0,"IF":0,"RPE":0},
          { "day": "Friday", "type": "Active Recovery", "details": "Coffee spin.", "minutes": 30,"TSS":0,"IF":0,"RPE":0},
          { "day": "Saturday", "type": "High Intensity", "details": "4 x 4m @ 105% FTP. Rest 4m between.", "minutes": 90,"TSS":0,"IF":0,"RPE":0},
          { "day": "Sunday", "type": "Long Endurance", "details": "Strict Zone 1.", "minutes": 150,"TSS":0,"IF":0,"RPE":0}
        ]
      },
       {
        "week_number": 2,
        "phase": "Phase 1: Base & Neuromuscular Activation",
        "focus": "Introduction to Sprint Load",
        "total_volume_estimate": "8-9 hours",
        "days": [
          { "day": "Monday", "type": "Rest", "details": "Complete Rest", "minutes": 0, "TSS":0,"IF":0,"RPE":0 },
          { "day": "Tuesday", "type": "High Intensity", "details": "6 x 10s MAX Sprints (Torque focus). Rest 5m between.", "minutes": 60,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Wednesday", "type": "Endurance", "details": "Zone 1 Steady.", "minutes": 90,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Thursday", "type": "Endurance", "details": "Zone 1 + 3 x 1m High Cadence (110rpm).", "minutes": 60,"TSS":0,"IF":0,"RPE":0},
          { "day": "Friday", "type": "Active Recovery", "details": "Coffee spin.", "minutes": 30,"TSS":0,"IF":0,"RPE":0},
          { "day": "Saturday", "type": "High Intensity", "details": "4 x 4m @ 105% FTP. Rest 4m between.", "minutes": 90,"TSS":0,"IF":0,"RPE":0},
          { "day": "Sunday", "type": "Long Endurance", "details": "Strict Zone 1.", "minutes": 150,"TSS":0,"IF":0,"RPE":0}
        ]
      ,
       {
        "week_number": 3,
        
        etcetera
    ]
  }
}
    Returns:
        str: Success message or error.
    """
    import json
    import sqlite3
    from datetime import datetime, timedelta
    
    session_id = session_id_ctx.get()
    user_id = None
    
    try:
        # 1. Resolve user_id
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row[0]
        
        # 1b. Check for existing plan
        if user_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                cursor = conn.execute("SELECT id FROM training_plans WHERE user_id = ?", (user_id,))
                if cursor.fetchone():
                    return "Error: You already have an active training plan. Impossible to create a new one."
        # 2. Parse JSON
        data = json.loads(plan_json)
        # Handle cases where the input is wrapped in markdown or is already a dict
        if isinstance(data, str):
             data = json.loads(data)
             
        plan_data = data.get("training_plan", data) # Support both wrapped and direct
        
        name = plan_data.get("name")
        goal = plan_data.get("goal")
        methodology = plan_data.get("methodology")
        start_date_str = plan_data.get("start_date")
        duration_weeks = plan_data.get("duration_weeks")
        schedule = plan_data.get("schedule", [])

        if not name or not start_date_str:
            return "Error: Plan must have a name and a start_date."

        # 3. Start Database Transaction
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            # Step A: Insert Plan
            cursor = conn.execute("""
                INSERT INTO training_plans (user_id, name, goal, methodology, start_date, duration_weeks)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, name, goal, methodology, start_date_str, duration_weeks))
            plan_id = cursor.lastrowid
            
            # Helper for date calculation
            start_dt = datetime.fromisoformat(start_date_str.split('T')[0])
            day_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            
            # Step B: Insert Weeks and Days
            for week in schedule:
                week_number = week.get("week_number")
                phase = week.get("phase")
                focus = week.get("focus")
                volume = week.get("total_volume_estimate")
                
                cursor = conn.execute("""
                    INSERT INTO training_plan_weeks (plan_id, week_number, phase, focus, total_volume_estimate)
                    VALUES (?, ?, ?, ?, ?)
                """, (plan_id, week_number, phase, focus, volume))
                week_id = cursor.lastrowid
                
                for day in week.get("days", []):
                    day_name = day.get("day")
                    workout_type = day.get("type")
                    details = day.get("details")
                    minutes = day.get("minutes", 0)
                    tss = day.get("TSS", 0)
                    intensity_factor = day.get("IF", 0.0)
                    rpe = day.get("RPE", 0)
                    
                    # Calculate planned_date
                    day_offset = day_map.get(day_name, 0)
                    planned_date = (start_dt + timedelta(weeks=(week_number-1), days=day_offset)).strftime("%Y-%m-%d")
                    
                    conn.execute("""
                        INSERT INTO training_plan_days (week_id, day_name, workout_type, details, minutes, tss, intensity_factor, rpe, planned_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (week_id, day_name, workout_type, details, minutes, tss, intensity_factor, rpe, planned_date))
            
            conn.commit()
            
        return f"Success: Training plan '{name}' (ID: {plan_id}) saved with {len(schedule)} weeks."

    except Exception as e:
        print(f"Error in save_training_plan: {str(e)}")
        return f"Error: {str(e)}"

def modify_training_plan_start_date(plan_id: int, new_start_date: str):
    """
    Modifies the start date of an existing training plan and updates all associated workout dates.
    
    Args:
        plan_id (int): The ID of the training plan to modify.
        new_start_date (str): The new start date in ISO format (e.g., "2024-03-30").
        
    Returns:
        str: Success message or error.
    """
    import sqlite3
    from datetime import datetime, timedelta
    
    try:
        new_start_dt = datetime.fromisoformat(new_start_date.split('T')[0])
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            
            # 1. Verify plan exists
            cursor = conn.execute("SELECT name FROM training_plans WHERE id = ?", (plan_id,))
            plan = cursor.fetchone()
            if not plan:
                return f"Error: Training plan with ID {plan_id} not found."
            
            # 2. Update training_plans table
            conn.execute("UPDATE training_plans SET start_date = ? WHERE id = ?", (new_start_date, plan_id))
            
            # 3. Update all associated days
            # Join training_plan_days and training_plan_weeks to get week_number and day_name
            cursor = conn.execute("""
                SELECT d.id, d.day_name, w.week_number 
                FROM training_plan_days d
                JOIN training_plan_weeks w ON d.week_id = w.id
                WHERE w.plan_id = ?
            """, (plan_id,))
            
            updates = []
            for row in cursor.fetchall():
                day_id = row['id']
                day_name = row['day_name']
                week_number = row['week_number']
                
                # Recalculate planned_date
                day_offset = day_map.get(day_name, 0)
                planned_date = (new_start_dt + timedelta(weeks=(week_number-1), days=day_offset)).strftime("%Y-%m-%d")
                updates.append((planned_date, day_id))
            
            # Batch update dates
            conn.executemany("UPDATE training_plan_days SET planned_date = ? WHERE id = ?", updates)
            
            conn.commit()
            
        return f"Success: Training plan '{plan['name']}' updated with new start date {new_start_date}."
        
    except Exception as e:
        print(f"Error in modify_training_plan_start_date: {str(e)}")
        return f"Error: {str(e)}"

def delete_training_plan():
    """
    Deletes the current user's training plan and all associated weeks, days, and workouts.
    
    Returns:
        str: Success message or error.
    """
    import sqlite3
    
    session_id = session_id_ctx.get()
    user_id = None
    
    try:
        # 1. Resolve user_id
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row[0]
        
        if not user_id:
            return "Error: Could not determine user ID from session. Please make sure you are logged in."

        # 2. Delete the plan
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.execute("DELETE FROM training_plans WHERE user_id = ?", (user_id,))
            if cursor.rowcount == 0:
                return "Error: No active training plan found for your user."
            conn.commit()
            
        return "Success: Your training plan and all associated data have been deleted."
        
    except Exception as e:
        print(f"Error in delete_training_plan: {str(e)}")
        return f"Error: {str(e)}"

def get_training_plan_week(plan_id: int, week_number: int):
    """
    Retrieves all details for a specific week of a training plan.
    
    Args:
        plan_id (int): The ID of the training plan.
        week_number (int): The week number to retrieve.
        
    Returns:
        str: JSON string of the week details.
    """
    import json
    import sqlite3
    
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            
            # 1. Get week info
            cursor = conn.execute("""
                SELECT * FROM training_plan_weeks 
                WHERE plan_id = ? AND week_number = ?
            """, (plan_id, week_number))
            week_row = cursor.fetchone()
            
            if not week_row:
                return f"Error: Week {week_number} not found for plan ID {plan_id}."
            
            week_data = dict(week_row)
            week_id = week_data['id']
            
            # 2. Get days for this week
            cursor = conn.execute("SELECT * FROM training_plan_days WHERE week_id = ? ORDER BY planned_date ASC", (week_id,))
            days = [dict(row) for row in cursor.fetchall()]
            
            week_data['days'] = days
            return json.dumps(week_data, indent=2)
            
    except Exception as e:
        print(f"Error in get_training_plan_week: {str(e)}")
        return f"Error: {str(e)}"

def get_intervals_icu_activities(oldest: str, newest: str):
    """
    Retrieves cycling activities from Intervals.icu for a specific time span.
    
    Args:
        oldest (str): The oldest date/time to retrieve (ISO format, e.g., "2025-12-15T00:00:00").
        newest (str): The newest date/time to retrieve (ISO format, e.g., "2025-12-16T00:00:00").
                      Max span: 14 days.
    
    Returns:
        str: A JSON-formatted string containing the filtered activities or an error message.
    """
    import json
    import requests
    from requests.auth import HTTPBasicAuth
    
    session_id = session_id_ctx.get()
    user_id = None
    athlete_id = None
    api_key = None
    
    try:
        # 1. Resolve user_id and Intervals.icu credentials
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row['user_id']
                
                if user_id:
                    cursor = conn.execute("SELECT intervals_athlete_id, intervals_api_key FROM users WHERE id = ?", (user_id,))
                    u_row = cursor.fetchone()
                    if u_row:
                        athlete_id = u_row['intervals_athlete_id']
                        api_key = u_row['intervals_api_key']

        if not athlete_id or not api_key:
            return "Error: Intervals.icu credentials not found for this user. Please configure them in settings."
        
        # 2. Call Intervals.icu API
        url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest={oldest}&newest={newest}"
        
        response = requests.get(
            url, 
            auth=HTTPBasicAuth('API_KEY', api_key),
            headers={'Accept': '*/*'}
        )
        
        if response.status_code != 200:
            return f"Error: Intervals.icu API returned status {response.status_code}: {response.text}"
        
        activities = response.json()
        
        # 3. Filter requested fields
        filtered_activities = []
        fields_to_keep = [
            "id", "start_date_local", "type", "icu_ctl", "icu_atl", 
            "icu_efficiency_factor", "decoupling", "icu_pm_ftp_watts", 
            "icu_zone_times", "icu_resting_hr", "carbs_used"
        ]
        
        for act in activities:
            filtered_act = {field: act.get(field) for field in fields_to_keep}
            filtered_activities.append(filtered_act)
            
        return json.dumps(filtered_activities, indent=2)
            
    except Exception as e:
        print(f"Error in get_intervals_icu_activities: {str(e)}")
        return f"Error: {str(e)}"

def get_training_plan_summary(plan_id: int = None):
    """
    Retrieves a summary of all training plans or a specific plan's metadata.
    
    Args:
        plan_id (int, optional): The ID of a specific plan to summarize. If None, returns all plans for the user.
        
    Returns:
        str: JSON string of the plan(s) summary.
    """
    import json
    import sqlite3
    
    session_id = session_id_ctx.get()
    user_id = None
    
    try:
        # 1. Resolve user_id
        if session_id:
            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    user_id = row[0]
                    
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            
            if plan_id:
                # Summary for a specific plan
                cursor = conn.execute("SELECT * FROM training_plans WHERE id = ?", (plan_id,))
                plan = cursor.fetchone()
                if not plan:
                    return f"Error: Plan ID {plan_id} not found."
                return json.dumps(dict(plan), indent=2)
            else:
                # List all plans for the user
                query = "SELECT id, name, goal, start_date, duration_weeks, created_at FROM training_plans WHERE user_id IS "
                params = []
                if user_id:
                    query += "? "
                    params.append(user_id)
                else:
                    query += "NULL "
                
                cursor = conn.execute(query, tuple(params))
                plans = [dict(row) for row in cursor.fetchall()]
                
                if not plans:
                    return "No training plans found."
                
                return json.dumps(plans, indent=2)
                
    except Exception as e:
        print(f"Error in get_training_plan_summary: {str(e)}")
        return f"Error: {str(e)}"

def query_training_plan_stats(plan_id: int, start_date: str = None, end_date: str = None):
    """
    Calculates aggregated statistics (TSS, minutes) for a training plan over a date range. 
    If no date range is specified, returns stats for the entire plan.
    
    Args:
        plan_id (int): The ID of the training plan.
        start_date (str, optional): Start of the date range (YYYY-MM-DD).
        end_date (str, optional): End of the date range (YYYY-MM-DD).
        
    Returns:
        str: JSON string of the aggregated statistics.
    """
    import json
    import sqlite3
    
    try:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT 
                    COUNT(*) as total_workouts,
                    SUM(minutes) as total_minutes,
                    SUM(tss) as total_tss,
                    AVG(intensity_factor) as avg_if,
                    AVG(rpe) as avg_rpe
                FROM training_plan_days d
                JOIN training_plan_weeks w ON d.week_id = w.id
                WHERE w.plan_id = ?
            """
            params = [plan_id]
            
            if start_date:
                query += " AND d.planned_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND d.planned_date <= ?"
                params.append(end_date)
                
            cursor = conn.execute(query, tuple(params))
            stats = cursor.fetchone()
            
            if not stats or stats['total_workouts'] == 0:
                return "No workouts found for the given criteria."
                
            return json.dumps(dict(stats), indent=2)
            
    except Exception as e:
        print(f"Error in query_training_plan_stats: {str(e)}")
        return f"Error: {str(e)}"

def calculator_tool(expr: str):
    """
    Use this tool for to evaluate simple math expressions. 
    args:
        expr (str): The math expression to evaluate.
    returns:
        str: The result of the evaluation.
    """

    try:
        return str(eval (expr, {"__builtins__": {}}))
    except Exception:
        return "Invalid expression"

def execute_calculation(code: str, file_path: str = None, custom_package: str = None, timeout: int = None): 
    """
    Executes Python code for calculations, logic, and text processing.
    Use this for math, data analysis, or data and string manipulation.
    Also takes a file path to upload to the execution environment.
    It is okay if the code creates output files in the execution environment. The files will be moved to an accessible location after the code is executed.
    Args:
        code (str): The Python code to execute.
        file_path (str): Optional file path to upload to the execution environment.
        custom_package (str): Optional custom package to install in the execution environment.
        timeout (int): Optional timeout in seconds.

    IMPORTANT:
    - This tool does NOT support plotting or image generation. Use 'generate_chart' for that.
    - The code must print the final result to stdout using `print()`.
    - You can import standard libraries (math, datetime, json, etc.) and numpy/pandas.
    - any files uploaded to the execution environment will be deleted after the code is executed.
    - any files uploaded to the execution environment will be in the root folder of the execution environment.

    Returns:
        str: the result of the calculation, could also include names of files created in the execution environment including a link to the file.
    """
    # Check execution mode
    execution_mode = execution_mode_ctx.get()
    e2b_api_key = os.getenv("E2B_API_KEY")

    if execution_mode == "e2b" and e2b_api_key:
        from e2b_code_interpreter import Sandbox
        try:
            # Get timeout
            if timeout is None:
                timeout = int(get_setting("e2b_timeout", "300"))
            else:
                timeout = int(timeout)
                print(f"Using custom sandbox timeout: {timeout}")
            with Sandbox.create() as sandbox:
                # check datafile was provided for the code to use.
                # if so, upload it to the sandbox
                if file_path:
                    # Sanitize input to prevent directory traversal
                    file_name = os.path.basename(file_path)
                    file_path = os.path.join("static/uploads", file_name)
                    print(f"Uploading file to E2B: {file_path}")
                    # Upload the dataset to the sandbox
                    dataset_path_in_sandbox = ""
                    with open(file_path, "rb") as f:
                        dataset_path_in_sandbox = sandbox.files.write(file_name, f) # Upload the file to the sandbox
                        print(f"File uploaded to E2B: {dataset_path_in_sandbox}")
                if custom_package:
                    # install the custom package
                    try:
                        print(f"Installing custom package {custom_package}")
                        sandbox.commands.run(f"pip install {custom_package}")
                    except Exception as e:
                        print(f"Error installing custom package {custom_package}: {str(e)}")
                        return f"Error installing custom package {custom_package}: {str(e)}"
                execution = sandbox.run_code(code, timeout=timeout)
                
                if execution.error:
                    print(f"Error in E2B Sandbox: {execution.error.name}: {execution.error.value}\n{execution.error.traceback}")
                    return f"Error in E2B Sandbox: {execution.error.name}: {execution.error.value}\n{execution.error.traceback}"
                
                output = ""
                if execution.logs.stdout:
                    output += "\n".join(execution.logs.stdout)
                if execution.logs.stderr:
                    output += "\nStdErr: " + "\n".join(execution.logs.stderr)
                
                # --- Introspection: Check for generated files ---
                # We run a script to list all files in the current directory
                # excluding hidden files, the uploaded file (if any), and common system files.
                excluded_files = [os.path.basename(file_path)] if file_path else []
                excluded_json = json.dumps(excluded_files)
                
                introspect_code = f"""
import os
import json

excluded = {excluded_json}
# Common system/env files to ignore
ignore_list = ['.', '..', '.bashrc', '.profile', '.bash_logout', '.local', '.cache', '.config', '.ipython', '.jupyter', '.npm', '.wget-hsts']

found_files = []
for f in os.listdir('.'):
    if f not in excluded and f not in ignore_list and not f.startswith('.'):
        if os.path.isfile(f):
            found_files.append(f)

print("FILES_JSON:" + json.dumps(found_files))
"""
                introspect_exec = sandbox.run_code(introspect_code, timeout=30)
                
                generated_files_links = []
                
                if introspect_exec.logs.stdout:
                    combined_out = "\n".join(introspect_exec.logs.stdout)
                    import re
                    match = re.search(r"FILES_JSON:(.*)", combined_out)
                    if match:
                        try:
                            file_list = json.loads(match.group(1))
                            if file_list:
                                print(f"Sandbox generated files: {file_list}")
                                
                                # Ensure generated directory exists
                                os.makedirs(GENERATED_DIR, exist_ok=True)
                                
                                # Download each file
                                for remote_filename in file_list:
                                    try:
                                        # Sanitize remote filename for local storage
                                        # Use uuid to prevent collisions
                                        ext = os.path.splitext(remote_filename)[1]
                                        local_filename = f"calc_{uuid.uuid4()}{ext}"
                                        dest_path = os.path.join(GENERATED_DIR, local_filename)
                                        
                                        # Download
                                        file_bytes = sandbox.files.read(remote_filename, format="bytes")
                                        with open(dest_path, "wb") as f:
                                            f.write(file_bytes)
                                            
                                        web_path = f"/static/generated/{local_filename}"
                                        
                                        # Log to DB
                                        session_id = session_id_ctx.get()
                                        if session_id:
                                            try:
                                                with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                                                    conn.execute("INSERT INTO generated_files (session_id, file_path) VALUES (?, ?)", (session_id, dest_path))
                                            except Exception as db_e:
                                                    print(f"Error saving generated file to DB: {db_e}")

                                        # Add to links
                                        generated_files_links.append(f"[{remote_filename}]({web_path})")
                                        
                                    except Exception as dl_e:
                                        print(f"Error downloading file {remote_filename}: {dl_e}")
                                        output += f"\nWarning: Could not download generated file {remote_filename}."
                        except json.JSONDecodeError:
                            print("Failed to parse introspection JSON")

                if not output and not generated_files_links:
                    return "Code executed successfully in Sandbox (no output)."
                
                result = output.strip()
                if generated_files_links:
                    result += "\n\n### Generated Files:\n"
                    for link in generated_files_links:
                        result += f"- {link}\n"
                
                return result
        except Exception as e:
            return f"Error executing code in E2B Sandbox: {str(e)}"

    # Local Execution (Default)
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

def generate_chart(code: str, file_path: str = None, custom_package: str = None, timeout: int = None):
    """
    Generates a chart or plot using Python and matplotlib.
    Use this tool WHENEVER the user asks for a visualization, graph, or chart.
    args:
        code (str): The Python code to execute.
        file_path (str): Optional file path to upload to the E2B sandbox.
        custom_package (str): Optional custom package to install in the E2B sandbox.
        timeout (int): Optional timeout in seconds.
    
    IMPORTANT:
    - You MUST use `plt.savefig('plot.png')` (or any filename ending in .png) to save the plot.
    - Do NOT use `plt.show()`.
    - The code should clear the figure before plotting: `plt.clf()` or `plt.close()`.
    - The tool will return the path to the generated image.
    - any files uploaded to the execution environment will be deleted after the code is executed.
    - any files uploaded to the execution environment will be in the root folder of the execution environment.

    Returns:
        str: JSON containing succes status and the path to the generated image.
    """
    # Check execution mode
    execution_mode = execution_mode_ctx.get()
    e2b_api_key = os.getenv("E2B_API_KEY")

    if execution_mode == "e2b" and e2b_api_key:
        from e2b_code_interpreter import Sandbox
        print("Generating chart in E2B Sandbox...")
        try:
            # Get timeout
            if timeout is None:
                timeout = int(get_setting("e2b_timeout", "300"))
            else:
                timeout = int(timeout)
                print(f"Using custom sandbox timeout: {timeout}")
            with Sandbox.create() as sandbox:
                if file_path:
                    # Sanitize input to prevent directory traversal
                    file_name = os.path.basename(file_path)
                    file_path = os.path.join("static/uploads", file_name)
                    print(f"Uploading file to E2B: {file_path}")
                    # Upload the dataset to the sandbox
                    dataset_path_in_sandbox = ""
                    with open(file_path, "rb") as f:
                        dataset_path_in_sandbox = sandbox.files.write(file_name, f) # Upload the file to the sandbox
                        print(f"File uploaded to E2B: {dataset_path_in_sandbox}")
                
                if custom_package:
                    # install the custom package
                    try:
                        print(f"Installing custom package {custom_package} in E2B Sandbox...")  
                        sandbox.commands.run(f"pip install {custom_package}")
                    except Exception as e:
                        print(f"Error installing custom package {custom_package}: {str(e)}")
                        return f"Error installing custom package {custom_package}: {str(e)}"
                
                execution = sandbox.run_code(code, timeout=timeout)
                if execution.error:
                    print(f"Error in E2B Sandbox: {execution.error.name}: {execution.error.value}")
                    return f"Error in E2B Sandbox: {execution.error.name}: {execution.error.value}"

                import base64
                generated_file = None
                output = ""
                if execution.logs.stdout:
                    output = "\n".join(execution.logs.stdout)

                for result in execution.results:
                    if result.png:
                        # Decode and save
                        img_data = base64.b64decode(result.png)
                        new_filename = f"chart_{uuid.uuid4()}.png"
                        dest_path = os.path.join(GENERATED_DIR, new_filename)
                        with open(dest_path, "wb") as f:
                            f.write(img_data)
                        generated_file = f"/static/generated/{new_filename}"
                        print(f"Retrieved chart from E2B: {dest_path}")
                        break
                
                if generated_file:
                     # Save to database if session_id is available
                    session_id = session_id_ctx.get()
                    if session_id:
                        try:
                            with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
                                conn.execute("INSERT INTO generated_files (session_id, file_path) VALUES (?, ?)", (session_id, dest_path))
                                print(f"Saved generated file record for session {session_id}: {dest_path}")
                        except Exception as db_e:
                            print(f"Error saving generated file to DB: {db_e}")

                    return json.dumps({
                        "status": "success",
                        "output": output,
                        "image_path": generated_file,
                        "message": "Chart generated successfully in Sandbox."
                    })
                else:
                     return json.dumps({
                        "status": "error",
                        "output": output,
                        "image_path": generated_file,
                        "message": "Code executed but no chart was generated."
                    })

        except Exception as e:
            return f"Error executing code in E2B Sandbox: {str(e)}"

    # Local Execution (Default)
    # Ensure generated directory exists
    os.makedirs(GENERATED_DIR, exist_ok=True)
    
    # Create a captured output stream
    output_capture = io.StringIO()
    print(output_capture)
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
                "message": "Chart generated successfully."
            })
        else:
            return json.dumps({
                "status": "error",
                "output": output,
                "image_path": generated_file,
                "message": "Code executed but no chart was generated."
            })
        
    except Exception as e:
        print(f"Error in generate_chart: {str(e)}")
        traceback.print_exc()
        return f"Error generating chart: {str(e)}"


def generate_plotly_chart(code: str, file_path: str = None, custom_package: str = None, timeout: int = None):
    """
    Generates an advanced chart using Python and Plotly.
    Use this tool for complex, interactive, or 3D visualizations that matplotlib cannot handle well.
    any files uploaded to the execution environment will be deleted after the code is executed.
    any files uploaded to the execution environment will be in the root folder of the execution environment.    

    args:
        code (str): The Python code to execute. 
        file_path (str): Optional file path to upload to the execution environment.
        custom_package (str): Optional custom package to install in the execution environment.
        timeout (int): Optional timeout in seconds.

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
    - After completing of this task, make sure you verify all the items of your execution plan have been completed. If not, continu to do so. 
    """
    # Check execution mode
    execution_mode = execution_mode_ctx.get()
    e2b_api_key = os.getenv("E2B_API_KEY")

    if execution_mode == "e2b" and e2b_api_key:
        from e2b_code_interpreter import Sandbox
        print("Generating Plotly chart in E2B Sandbox...")
        try:
            # Get timeout
            if timeout is None:
                timeout = int(get_setting("e2b_timeout", "300"))
            else:
                timeout = int(timeout)
                print(f"Using custom sandbox timeout: {timeout}")
            
            with Sandbox.create() as sandbox:
                # Install kaleido for PNG export (version 0.2.1 is self-contained)
                sandbox.commands.run("pip install kaleido==0.2.1")

                if file_path:
                    # Sanitize input to prevent directory traversal
                    file_name = os.path.basename(file_path)
                    file_path = os.path.join("static/uploads", file_name)
                    print(f"Uploading file to E2B: {file_path}")
                    # Upload the dataset to the sandbox
                    dataset_path_in_sandbox = ""
                    with open(file_path, "rb") as f:
                        dataset_path_in_sandbox = sandbox.files.write(file_name, f) # Upload the file to the sandbox
                        print(f"File uploaded to E2B: {dataset_path_in_sandbox}")
                
                if custom_package:
                    # install the custom package
                    try:
                        print(f"Installing custom package {custom_package} in E2B Sandbox...")
                        sandbox.commands.run(f"pip install {custom_package}")
                    except Exception as e:
                        print(f"Error installing custom package {custom_package}: {str(e)}")
                        return f"Error installing custom package {custom_package}: {str(e)}"
                # 1. Run User Code
                execution = sandbox.run_code(code, timeout=timeout)
                if execution.error:
                    print(f"Error in E2B Sandbox: {execution.error.name}: {execution.error.value}")
                    return f"Error in E2B Sandbox: {execution.error.name}: {execution.error.value}"

                output = ""
                if execution.logs.stdout:
                    output = "\n".join(execution.logs.stdout)
                
                # 2. Introspection and Save
                introspect_code = """
import os
import uuid
import json

generated_files = {}

if 'fig' in locals():
    # Generate unique names
    base_name = str(uuid.uuid4())
    html_name = f"plotly_{base_name}.html"
    png_name = f"plotly_{base_name}.png"
    
    try:
        fig.write_html(html_name)
        generated_files['html'] = html_name
    except Exception as e:
        print(f"Error saving HTML: {e}")
        
    try:
        # write_image requires kaleido or similar installed in env
        # If it fails, we skipping PNG
        fig.write_image(png_name)
        generated_files['png'] = png_name
    except Exception as e:
        print(f"Error saving PNG: {e}")

print("JSON_RESULT:" + json.dumps(generated_files))
"""
                save_exec = sandbox.run_code(introspect_code, timeout=60)
                if save_exec.error:
                     print(f"Introspection failed: {save_exec.error}")
                
                # Parse stdout for JSON result
                combined_stdout = "\n".join(save_exec.logs.stdout)
                import re
                match = re.search(r"JSON_RESULT:(.*)", combined_stdout)
                
                generated_png = None
                generated_html = None
                
                if match:
                    json_str = match.group(1)
                    try:
                        files_map = json.loads(json_str)
                        
                        if 'html' in files_map:
                            remote_path = files_map['html']
                            local_filename = f"e2b_{remote_path}"
                            dest_path = os.path.join(GENERATED_DIR, local_filename)
                            
                            # Download
                            file_bytes = sandbox.files.read(remote_path, format="bytes")
                            with open(dest_path, "wb") as f:
                                f.write(file_bytes)
                            
                            generated_html = f"/static/generated/{local_filename}"
                            
                        if 'png' in files_map:
                            remote_path = files_map['png']
                            local_filename = f"e2b_{remote_path}"
                            dest_path = os.path.join(GENERATED_DIR, local_filename)
                            
                            # Download
                            file_bytes = sandbox.files.read(remote_path, format="bytes")
                            with open(dest_path, "wb") as f:
                                f.write(file_bytes)
                                
                            generated_png = f"/static/generated/{local_filename}"
                            
                    except Exception as e:
                         print(f"Error parsing/downloading files: {e}")

                # Save to database
                if generated_png or generated_html:
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
                    "message": "Plotly chart generated successfully in E2B."
                }
                
                if generated_png:
                    response_data["image_path"] = generated_png
                
                if generated_html:
                    response_data["html_path"] = generated_html
                    link_msg = f'Interactive Plotly chart generated. <a href="{generated_html}" onclick="event.stopPropagation(); event.preventDefault(); window.open(this.href, \'_blank\'); return false;" class="text-blue-500 hover:underline">🔗 Open chart in new tab</a>'
                    response_data["message"] =  link_msg

                return json.dumps(response_data)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "output": output,
                "image_path": generated_file,
                "message": "Code executed but no chart was generated."
            })

    # Local Execution (Default)
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
            return json.dumps({
                "status": "error",
                "output": output,
                "image_path": generated_file,
                "message": "Code executed but no chart was saved. Did you assign the figure to `fig`?"
            })
        
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
    args:
    query: The query to send to Wolfram Alpha.

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
    DO NOT use this for data visualizations, maps, charts, or graphs - use generate_chart or generate_plotly_chart instead.
    
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
    Use this tool when you need coordinates or location data to answer a users question.
    args:
        location (str): The location to get coordinates for.
    returns:
        str: The latitude and longitude of the location.
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

# def get_weather(location: str):
#     """
#     Gets current weather information for a specific location using XWeather MCP server.
#     Use this tool when the user asks about weather, temperature, conditions,  etc.
#     
#     Args:
#         location (str): The location to get weather for (city name, address, etc.)
#     
#     Returns:
#         str: Current weather information for the location
#     """
#     try:
#         import requests
#         
#         # Get XWeather credentials from environment
#         xweather_id = os.getenv("XWEATHER_MCP_ID")
#         xweather_secret = os.getenv("XWEATHER_MCP_SECRET")
#         
#         if not xweather_id or not xweather_secret:
#             return "Error: XWeather API credentials not configured. Please set XWEATHER_MCP_ID and XWEATHER_MCP_SECRET in .env file."
#         
#         # MCP server endpoint
#         mcp_url = "https://mcp.api.xweather.com/mcp"
#         
#         # Prepare authorization (using client_id:client_secret format)
#         auth_string = f"{xweather_id}_{xweather_secret}"
#         
#         # Make request to MCP server
#         # The MCP server expects a JSON-RPC 2.0 request
#         payload = {
#             "jsonrpc": "2.0",
#             "id": 1,
#             "method": "tools/call",
#             "params": {
#                 "name": "xweather_get_current_weather",
#                 "arguments": {
#                     "location": location
#                 }
#             }
#         }
#         
#         headers = {
#             "Authorization": f"Bearer {auth_string}",
#             "Content-Type": "application/json",
#             "Accept": "application/json, text/event-stream"
#         }
#         
#         response = requests.post(mcp_url, json=payload, headers=headers, timeout=10)
#         
#         print(f"XWeather API Status: {response.status_code}")
#         print(f"XWeather API Response: {response.text[:500]}")  # Print first 500 chars
#         
#         if response.status_code == 200:
#             try:
#                 # The response is in SSE format, need to parse it
#                 response_text = response.text
#                 
#                 # Extract JSON from SSE data field
#                 if "data:" in response_text:
#                     # Split by lines and find the data line
#                     for line in response_text.split('\n'):
#                         if line.startswith('data:'):
#                             json_str = line[5:].strip()  # Remove 'data:' prefix
#                             data = json.loads(json_str)
#                             
#                             # Check for errors in the result
#                             if "result" in data:
#                                 result = data["result"]
#                                 if isinstance(result, dict) and result.get("isError"):
#                                     # Extract error message
#                                     if "content" in result and len(result["content"]) > 0:
#                                         error_text = result["content"][0].get("text", "Unknown error")
#                                         # Add helpful hint for invalid location errors
#                                         if "invalid" in error_text.lower() and "location" in error_text.lower():
#                                             error_text += "\n\nHint: Try specifying the location more precisely, e.g., 'Minneapolis, MN', 'London, UK', or use a zip code like '55401'."
#                                         return f"Weather service error: {error_text}"
#                                     return f"Weather service error: {result}"
#                                 
#                                 # Return successful result
#                                 if "content" in result and len(result["content"]) > 0:
#                                     return result["content"][0].get("text", str(result))
#                                 return str(result)
#                             else:
#                                 return str(data)
#                     
#                     return f"Error: Could not parse SSE response: {response_text[:200]}"
#                 else:
#                     # Try parsing as regular JSON
#                     data = response.json()
#                     if "result" in data:
#                         return str(data["result"])
#                     else:
#                         return str(data)
#                         
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error: {e}")
#                 print(f"Response content: {response.text}")
#                 return f"Error: Received invalid JSON from XWeather API. Response: {response.text[:200]}"
#         else:
#             print(response.text)
#             return f"Error: XWeather API returned status {response.status_code}: {response.text}"
#             
#     except Exception as e:
#         return f"Error getting weather: {str(e)}"

# def get_forecast_weather(location: str):
#     """
#     Gets multi-day weather forecast for a specific location using XWeather MCP server.
#     Use this tool for planning, event scheduling, travel preparation, or when the user asks about future weather.
#     
#     Args:
#         location (str): The location to get forecast for (city, state/country, or zip code)
#     
#     Returns:
#         str: Multi-day weather forecast with temperatures, precipitation, wind, and conditions
#     """
#     try:
#         import requests
#         
#         xweather_id = os.getenv("XWEATHER_MCP_ID")
#         xweather_secret = os.getenv("XWEATHER_MCP_SECRET")
#         
#         if not xweather_id or not xweather_secret:
#             return "Error: XWeather API credentials not configured."
#         
#         mcp_url = "https://mcp.api.xweather.com/mcp"
#         auth_string = f"{xweather_id}_{xweather_secret}"
#         
#         payload = {
#             "jsonrpc": "2.0",
#             "id": 1,
#             "method": "tools/call",
#             "params": {
#                 "name": "xweather_get_forecast_weather",
#                 "arguments": {
#                     "location": location
#                 }
#             }
#         }
#         
#         headers = {
#             "Authorization": f"Bearer {auth_string}",
#             "Content-Type": "application/json",
#             "Accept": "application/json, text/event-stream"
#         }
#         
#         response = requests.post(mcp_url, json=payload, headers=headers, timeout=10)
#         
#         if response.status_code == 200:
#             response_text = response.text
#             if "data:" in response_text:
#                 for line in response_text.split('\n'):
#                     if line.startswith('data:'):
#                         json_str = line[5:].strip()
#                         try:
#                             data = json.loads(json_str)
#                             
#                             if "result" in data:
#                                 result = data["result"]
#                                 if isinstance(result, dict) and result.get("isError"):
#                                     if "content" in result and len(result["content"]) > 0:
#                                         error_text = result["content"][0].get("text", "Unknown error")
#                                         if "invalid" in error_text.lower() and "location" in error_text.lower():
#                                             error_text += "\n\nHint: Try specifying the location more precisely, e.g., 'Minneapolis, MN', 'London, UK', or use a zip code."
#                                         return f"Weather service error: {error_text}"
#                                 
#                                 if "content" in result and len(result["content"]) > 0:
#                                     return result["content"][0].get("text", str(result))
#                                 return str(result)
#                         except json.JSONDecodeError:
#                             pass
#                             
#                 return f"Error: Could not parse forecast response"
#         else:
#             return f"Error: XWeather API returned status {response.status_code}"
#             
#     except Exception as e:
#         return f"Error getting forecast: {str(e)}"

# def get_precipitation_timing(location: str):
#     """
#     Gets timing information for upcoming precipitation (rain, snow, etc.) in the next hours to days.
#     Use this tool when the user asks "when will it rain?", "when will it stop raining?", or needs precipitation timing for outdoor activities.
#     
#     Args:
#         location (str): The location to get precipitation timing for (city, state/country, or zip code)
#     
#     Returns:
#         str: Start/stop timing and duration of upcoming precipitation
#     """
#     try:
#         import requests
#         
#         xweather_id = os.getenv("XWEATHER_MCP_ID")
#         xweather_secret = os.getenv("XWEATHER_MCP_SECRET")
#         
#         if not xweather_id or not xweather_secret:
#             return "Error: XWeather API credentials not configured."
#         
#         mcp_url = "https://mcp.api.xweather.com/mcp"
#         auth_string = f"{xweather_id}_{xweather_secret}"
#         
#         payload = {
#             "jsonrpc": "2.0",
#             "id": 1,
#             "method": "tools/call",
#             "params": {
#                 "name": "xweather_get_forecast_precipitation_timing",
#                 "arguments": {
#                     "location": location
#                 }
#             }
#         }
#         
#         headers = {
#             "Authorization": f"Bearer {auth_string}",
#             "Content-Type": "application/json",
#             "Accept": "application/json, text/event-stream"
#         }
#         
#         response = requests.post(mcp_url, json=payload, headers=headers, timeout=10)
#         
#         if response.status_code == 200:
#             response_text = response.text
#             if "data:" in response_text:
#                 for line in response_text.split('\n'):
#                     if line.startswith('data:'):
#                         json_str = line[5:].strip()
#                         data = json.loads(json_str)
#                         
#                         if "result" in data:
#                             result = data["result"]
#                             if isinstance(result, dict) and result.get("isError"):
#                                 if "content" in result and len(result["content"]) > 0:
#                                     error_text = result["content"][0].get("text", "Unknown error")
#                                     if "invalid" in error_text.lower() and "location" in error_text.lower():
#                                         error_text += "\n\nHint: Try specifying the location more precisely, e.g., 'Minneapolis, MN', 'London, UK', or use a zip code."
#                                     return f"Weather service error: {error_text}"
#                             
#                             if "content" in result and len(result["content"]) > 0:
#                                 return result["content"][0].get("text", str(result))
#                             return str(result)
#                         
#                 return f"Error: Could not parse precipitation timing response"
#         else:
#             return f"Error: XWeather API returned status {response.status_code}"
#             
#     except Exception as e:
#         return f"Error getting precipitation timing: {str(e)}"

def get_open_meteo_weather(location: str):
    """
    Gets current weather and forecast for a specific location using the Open-Meteo API.
    Use this tool when the user asks about weather, temperature, wind, humidity, or forecasts.
    
    Args:
        location (str): The location to get weather for (city name, address, etc.)
    
    Returns:
        str: Weather information including current conditions, next 24 hours, and 7-day daily forecast.
    """
    try:
        import requests
        from geopy.geocoders import Nominatim
        
        # 1. Geocode location
        geolocator = Nominatim(user_agent="htmx_chatbot")
        loc = geolocator.geocode(location)
        if not loc:
            return f"Error: Could not find coordinates for '{location}'."
        
        lat = loc.latitude
        lon = loc.longitude
        
        # 2. Call Open-Meteo API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"Error: Open-Meteo API returned status {response.status_code}: {response.text}"
        
        data = response.json()
        
        # WMO Weather interpretation codes (WW)
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm: Slight or moderate", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        
        # 3. Format result
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        wind = current.get("wind_speed_10m")
        humidity = current.get("relative_humidity_2m")
        w_code = current.get("weather_code")
        condition = weather_codes.get(w_code, "Unknown")
        
        unit_temp = data.get("current_units", {}).get("temperature_2m", "°C")
        unit_wind = data.get("current_units", {}).get("wind_speed_10m", "km/h")
        
        result = f"Weather for {loc.address}:\n"
        result += f"Current: {condition}, {temp}{unit_temp}, Wind: {wind}{unit_wind}, Humidity: {humidity}%\n"
        
        # Hourly Forecast (Next 24 hours)
        hourly = data.get("hourly", {})
        h_times = hourly.get("time", [])[:24]
        h_temps = hourly.get("temperature_2m", [])[:24]
        h_codes = hourly.get("weather_code", [])[:24]
        
        if h_times:
            result += "\nHourly Forecast (Next 24 hours):\n"
            # Show every 3rd hour to keep it concise
            for i in range(0, len(h_times), 3):
                t = h_times[i].split("T")[-1]
                cond = weather_codes.get(h_codes[i], "")
                result += f"- {t}: {h_temps[i]}{unit_temp} ({cond})\n"
        
        # Daily Forecast (7 days)
        daily = data.get("daily", {})
        d_times = daily.get("time", [])
        d_max = daily.get("temperature_2m_max", [])
        d_min = daily.get("temperature_2m_min", [])
        d_precip = daily.get("precipitation_sum", [])
        d_codes = daily.get("weather_code", [])
        
        if d_times:
            result += "\n7-Day Daily Forecast:\n"
            for i in range(len(d_times)):
                date = d_times[i]
                cond = weather_codes.get(d_codes[i], "")
                result += f"- {date}: {cond}, High: {d_max[i]}{unit_temp}, Low: {d_min[i]}{unit_temp}, Precip: {d_precip[i]}mm\n"
        
        return result
        
    except Exception as e:
        return f"Error getting weather from Open-Meteo: {str(e)}"



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
async def post_chat(request: Request, prompt: str = Form(...), file: UploadFile = File(None), user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    
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
        save_message(session_id, "user", prompt, image_path, mime_type, user_id=user_id)

    stream_id = f"msg-{uuid.uuid4()}"
    
    # Render User Message
    from datetime import datetime
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message_html = render_user_message(prompt, image_path, now_ts)
    
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
    <div id="{stream_id}" class="flex justify-start mb-4 animate-fade-in">
        <div id="{stream_id}-content" 
             hx-ext="sse" 
             sse-connect="{stream_url}" 
             sse-swap="message" 
             hx-swap="beforeend"
             class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
            <span id="cursor" class="inline-block w-2 h-5 bg-blue-500 cursor-blink align-middle"></span>
        </div>
    </div>
    """
    
    return user_message_html + bot_placeholder_html

@app.get("/stream")
async def stream_response(request: Request, prompt: str, session_id: str = Cookie(None), stream_id: str = Query(...), image_path: Optional[str] = None, mime_type: Optional[str] = None, selected_model: str = Cookie(None), execution_mode: str = Cookie(None)):
    """
    The 'session_id' is automatically extracted from the browser cookie.
    """
    # Set client IP in context for tools to use
    if request.client:
        client_ip_ctx.set(request.client.host)
    
    user_settings = None
    user_id = None
    coach_mode_enabled = False # Default
    automatic_function_calling = True # Default
    if session_id:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            cursor = conn.execute("SELECT user_id FROM conversations WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                user_id = row[0]
                user_settings = get_user_settings(user_id)
                if user_settings:
                    coach_mode_enabled = user_settings.get('coach_mode_enabled', False)
                    automatic_function_calling = user_settings.get('automatic_function_calling', True)
    # Fallback/Override logic
    if selected_model is None:
        selected_model = user_settings['selected_model'] if user_settings else get_setting("selected_model", "gemini-2.5-flash")
    if execution_mode is None:
        execution_mode = get_setting("execution_mode", "local")

    # Set execution mode
    execution_mode_ctx.set(execution_mode)
    
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
                
                # Determine if we should upload to Provider (Images/Video) or just provide path (Data)
                is_media = mime_type and (mime_type.startswith("image/") or mime_type.startswith("video/"))
                
                if is_media:
                    try:
                        # Upload to Provider
                        print(f"Uploading file to Provider: {local_path} ({mime_type})")
                        if llm_provider:
                            uploaded_file = await llm_provider.upload_file(local_path, mime_type=mime_type, wait_for_active=True)
                        else:
                            raise ValueError("LLM Provider not initialized")
                        
                        # Handle different return types (Gemini vs OpenAI)
                        if hasattr(uploaded_file, "state") and uploaded_file.state.name == "FAILED":
                             yield format_sse("<div><strong>Error:</strong> File processing failed.</div>")
                             return
                        elif isinstance(uploaded_file, dict) and "error" in uploaded_file:
                             yield format_sse(f"<div><strong>Error:</strong> {uploaded_file['error']}</div>")
                             return

                        current_parts.append(uploaded_file)
                        
                        # Log URI if available
                        uri_log = getattr(uploaded_file, "uri", "local_file_only")
                        print(f"File uploaded successfully: {uri_log}")
                        
                        # Update the message in DB with the Gemini URI
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
                                    (getattr(uploaded_file, "uri", None), session_id, image_path)
                                )
                        
                    except Exception as e:
                        print(f"Error uploading file: {e}")
                        yield format_sse(f"<div><strong>Error uploading file:</strong> {str(e)}</div>")
                        return
                else:
                    # Data File handling
                    print(f"File is data (not media). Skipping Provider upload: {local_path}")
                    # Provide system notification about the file
                    file_name = os.path.basename(local_path)
                    current_parts.append(types.Part(text=f"[System Notification: User uploaded file '{file_name}' to '{local_path}'. You can use tools like 'read_uploaded_file' to read it, or pass the path to 'execute_calculation', 'generate_chart' or 'generate_plotly_chart'.]"))

            messages_payload.append({
                "role": "user",
                "parts": current_parts
            })

            # 2. Call Gemini
            # Prepare the model's tools and system instruction
            tools = []
            current_instruction = get_system_instruction(user_id) # Start with base instruction
            has_file_in_context = uploaded_file is not None # Check if file was uploaded in this turn
            current_instruction += f"\n\n{get_current_datetime()}, use this for the duration of this turn" 
            
            # Inject User Memories
            if user_id:
                memories = get_user_memories(user_id)
                if memories:
                    memories_text = "\n".join([f"- {m['content']}" for m in memories])
                    current_instruction += f"\n\n### User Memories (Long-term Context)\nThese are things you've learned about the user in the past:\n{memories_text}"
            
            # Check logic:
            # If we represent a "Vision" turn (uploading image/video), we disable tools and keep history (Context).
            # If we represent a "Text/Tool" turn (uploading data or just text), we MUST ENABLE tools.
            # However, Gemini throws 400 if Tools + Images exist in history.
            # So, for Tool turns, we sanitize history by removing media parts.
            
            if not has_file_in_context:
                # We are in Tool Mode (no current media upload)
                sanitized_msgs = []
                history_filtered = False
                
                for msg in messages_payload:
                    clean_parts = []
                    for part in msg.get('parts', []):
                        is_media = False
                        if hasattr(part, 'file_data') and part.file_data:
                            mime = getattr(part.file_data, 'mime_type', '')
                            if mime.startswith('image/') or mime.startswith('video/'):
                                is_media = True
                        
                        if not is_media:
                            clean_parts.append(part)
                        else:
                            history_filtered = True
                    
                    if clean_parts:
                        sanitized_msgs.append({"role": msg["role"], "parts": clean_parts})
                
                if history_filtered:
                    messages_payload = sanitized_msgs

            if not has_file_in_context:
                if tavily_client:
                    tools.append(search_web)
                    current_instruction += """\n\n- You have access to a 'search_web' tool. You must use it whenever the user asks for current information, news, or facts you don't know. 
                    Do NOT invent new tools. Only use 'search_web' for searching. you can specify the number of results (max 10) with the 'numberofwebsearchresults' parameter."""
                    
                    tools.append(crawl_website)
                    current_instruction += """\n\n- You have access to a 'crawl_website' tool. 
                    Use it when you need to gather content from multiple pages of a specific website, explore a site's structure, or extract information from specific URLs. 
                    Parameters: url (required), max_depth (default 2), limit (default 10), instructions (optional filter)."""
                
                tools.append(read_source_code)
                current_instruction += """\n\n- You have access to a 'read_source_code' tool. 
                Use it for reading your own source code files. It returns TEXT output. It CANNOT generate images."""
                
                tools.append(read_uploaded_file)
                current_instruction += """\n\n- You have access to a 'read_uploaded_file' tool. 
                Use it to see the text content of files that were previously uploaded to the uploads folder. It returns the raw text content."""

                tools.append(write_source_code)
                current_instruction += """\n\n- You have access to a 'write_source_code' tool. Use it for writing (modified) source code to file. 
                Leave session_id empty in the function call, this is automatically taken from the environment variable.
                When providing download links for files created by the 'write_source_code' tool, ALWAYS use the relative web path starting with '/' (e.g., '/static/generated_code/filename.ext'). NEVER prefix the URL with 'sandbox:', 'file://', or any other scheme.
                It returns TEXT output. It CANNOT generate images."""

                if pisa:
                    tools.append(convert_md_to_pdf)
                    current_instruction += """\n\n- You have access to a 'convert_md_to_pdf' tool.
                    Use it to convert Markdown text to a PDF file. It returns a URL to the generated PDF.
                    Always provide the link to the user so they can download it.
                    """
                
                tools.append(rename_chat_tool)
                current_instruction += """\n\n- You have access to a 'rename_chat_tool'.
                Use it to rename the current chat session or a specific session ID.
                """

                tools.append(archive_chat_tool)
                current_instruction += """\n\n- You have access to a 'archive_chat_tool'.
                Use it to archive the current chat session.
                """

                tools.append(delete_chat_tool)
                current_instruction += """\n\n- You have access to a 'delete_chat_tool'.
                Use it to permanently delete a chat session.
                """
               
                if(get_setting("execution_mode", "") != "e2b"):
                    tools.append(import_package)
                    current_instruction += """\n\n- You have access to an 'import_package' tool. 
                Use it for checking if a package is installed (install=False) or installing and importing directly (install=True). It returns TEXT output.  
                It CANNOT generate images. It can also just check if a package is installed."""
                    
                
                default_timeout = int(get_setting("e2b_timeout", "300"))
                tools.append(execute_calculation)
                current_instruction += """\n\n- You have access to an 'execute_calculation' tool. 
                - Use it for math, logic, text processing, or data analysis (numpy/pandas) or any arbitray python code executions. 
                - It's fine for the code to generate output file(s). If the code creates output files in the execution environment, the files will be moved to an accessible location after the code is executed.
                - If a file is needed to be processed by the code, the code must refer to only the filename (not the full path). 
                - you can provide execute_calculation tool with the name of module/package to install and import.
                - You can also provide a timeout in seconds to the execute_calculation tool ONLY if you expect the code to take longer than the default timeout of {default_timeout} seconds.
                -It returns TEXT output. It CANNOT generate images. DO NOT provide Python code to the user - ALWAYS call the tool."""

                tools.append(generate_chart)
                current_instruction += """\n\n- You have access to a 'generate_chart' tool using matplotlib. 
                - Use this for standard charts (bar, line, pie, scatter, histograms). 
                - If a file is needed to be processed by the code, the code must refer to only the filename (not the full path). 
                - you can provide generate_chart tool with the name of module/package to install and import.
                - You can also provide a timeout in seconds to the generate_chart tool ONLY if you expect the code to take longer than the default timeout of {default_timeout} seconds.
                - It returns JSON containing the path to the generated chart (image file) (if successful). 
                - DO NOT provide Python code to the user - ALWAYS call the tool."""
                
                tools.append(generate_plotly_chart)
                current_instruction += """\n\n- You have access to a 'generate_plotly_chart' tool using Plotly. 
                - Use this for ADVANCED visualizations: 3D plots, interactive charts, sunburst/treemap, Sankey diagrams, animated charts, geographic maps. 
                - If a file is needed to be processed by the code, the code must refer to only the filename (not the full path). 
                - you can provide generate_plotly_chart tool with the name of module/package to install and import.
                - You can also provide a timeout in seconds to the generate_plotly_chart tool ONLY if you expect the code to take longer than the default timeout of {default_timeout} seconds.
                - It returns JSON containing the path to the generated chart (image file) and/or the generated html file (if successful). 
                - DO NOT provide Python code to the user - ALWAYS call the tool. Use this when matplotlib's generate_chart cannot handle the request."""

                tools.append(wolfram_alpha_query)
                current_instruction += """\n\n- You have access to a 'wolfram_alpha_query' tool. 
                - Use it for complex math, scientific data, unit conversions, and factual queries that might not be in your knowledge base.
                - When the tool returns images or links to images contemplate using these in your final response.
                - GUIDELINES:
                - Convert inputs to simplified keyword queries (e.g. "France population" instead of "how many people live in France").
                - Send queries in English only.
                - Use proper Markdown for math formulas: '$$...$$' for block formulas
                - Use named physical constants (e.g., 'speed of light') without numerical substitution.
                - Include a space between compound units (e.g., "m Ω" instead of "mΩ").
                - If data for multiple properties is needed, make separate calls for each property.
                """

                #tools.append(generate_image)
                #current_instruction += """\n\n- You have access to a 'generate_image' tool.
                #Use it for artistic or creative image requests like 'draw a cat', 'create a sunset landscape', etc. 
                #DO NOT use this for data visualizations - use generate_chart instead."""

                #tools.append(get_current_datetime)
                #current_instruction +="""\n\n- You have access to a 'get_current_datetime' tool. 
                #- Use it when asked about the current time or date. It automatically detects the user's timezone from their IP. 
                #- You can also pass a 'timezone' argument (e.g., 'Asia/Tokyo') if the user explicitly requests a specific timezone."""

                tools.append(get_user_timezone)
                current_instruction += """\n\n- You have access to a 'get_user_timezone' tool. 
                - Use it when the user asks 'What is my timezone?' or wants to know the timezone detected from their IP."""

                tools.append(get_coordinates)
                current_instruction += """\n\n- You have access to a 'get_coordinates' tool. 
                - Use it to find the latitude and longitude of locations."""

                tools.append(get_open_meteo_weather)
                current_instruction += """\n\n- You have access to a 'get_open_meteo_weather' tool. 
                Use it when the user asks about current weather, temperature, wind, humidity, or forecasts for any location. It uses the Open-Meteo API."""

                
                tools.append(add_user_memory)
                current_instruction += """\n\n- You have access to an 'add_user_memory' tool. 
                - Use it to save important information, preferences, or facts about the user that should persist across chat sessions.
                - Only save information that is actually useful for long-term context."""
                
                if coach_mode_enabled:

                    current_instruction += """\n\n- you are an experienced cycling performance specialist and coach. 
                    I want you to help me create a balanced training plan. Once that training plans is created, 
                    I want you to help me execute it.

                    CRITICAL PROTOCOL - EXISTING PLANS: Before generating any new workout files, 
                    creating new schedules, or answering questions about "my workout" or "my plan," you MUST first execute the get_training_plan_summary tool.
                     * If a plan returns: Use that specific plan's data (and get_training_plan_week if needed) to answer the request. 
                     * If no plan returns: Only then proceed to design a new one or create ad-hoc workouts. 
                     * Never assume the user does not have a plan without checking the database first.
You need to ask me (interactively) questions so you have enough information to create the plan.
Plans have a maximum duration of 18 weeks.
I want the output structured like this EXAMPLE:
{
  "training_plan": {
    "name": "12-Week Polarized: Endurance & Peak Power (Weekly Detail)",
    "goal": "Build deep aerobic base while maximizing neuromuscular power output.",
    "methodology": "Polarized (80/20)",
    "start_date": "2025-12-23",
    "duration_weeks": 12,
    "schedule": [
      {
        "week_number": 1,
        "phase": "Phase 1: Base & Neuromuscular Activation",
        "focus": "Introduction to Sprint Load",
        "total_volume_estimate": "8-9 hours",
        "days": [
          { "day": "Monday", "type": "Rest", "details": "Complete Rest", "minutes": 0, "TSS":0,"IF":0,"RPE":0 },
          { "day": "Tuesday", "type": "High Intensity", "details": "6 x 10s MAX Sprints (Torque focus). Rest 5m between.", "minutes": 60,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Wednesday", "type": "Endurance", "details": "Zone 1 Steady.", "minutes": 90,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Thursday", "type": "Endurance", "details": "Zone 1 + 3 x 1m High Cadence (110rpm).", "minutes": 60,"TSS":0,"IF":0,"RPE":0},
          { "day": "Friday", "type": "Active Recovery", "details": "Coffee spin.", "minutes": 30,"TSS":0,"IF":0,"RPE":0},
          { "day": "Saturday", "type": "High Intensity", "details": "4 x 4m @ 105% FTP. Rest 4m between.", "minutes": 90,"TSS":0,"IF":0,"RPE":0},
          { "day": "Sunday", "type": "Long Endurance", "details": "Strict Zone 1.", "minutes": 150,"TSS":0,"IF":0,"RPE":0}
        ]
      },
       {
        "week_number": 2,
        "phase": "Phase 1: Base & Neuromuscular Activation",
        "focus": "Introduction to Sprint Load",
        "total_volume_estimate": "8-9 hours",
        "days": [
          { "day": "Monday", "type": "Rest", "details": "Complete Rest", "minutes": 0, "TSS":0,"IF":0,"RPE":0 },
          { "day": "Tuesday", "type": "High Intensity", "details": "6 x 10s MAX Sprints (Torque focus). Rest 5m between.", "minutes": 60,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Wednesday", "type": "Endurance", "details": "Zone 1 Steady.", "minutes": 90,"TSS":0,"IF":0,"RPE":0 },
          { "day": "Thursday", "type": "Endurance", "details": "Zone 1 + 3 x 1m High Cadence (110rpm).", "minutes": 60,"TSS":0,"IF":0,"RPE":0},
          { "day": "Friday", "type": "Active Recovery", "details": "Coffee spin.", "minutes": 30,"TSS":0,"IF":0,"RPE":0},
          { "day": "Saturday", "type": "High Intensity", "details": "4 x 4m @ 105% FTP. Rest 4m between.", "minutes": 90,"TSS":0,"IF":0,"RPE":0},
          { "day": "Sunday", "type": "Long Endurance", "details": "Strict Zone 1.", "minutes": 150,"TSS":0,"IF":0,"RPE":0}
        ]
      ,
       {
        "week_number": 3,
        
        etcetera
    ]
  }
}

after creation of a training plan, add a note to the user's memory remember the creation date and name of the plan.

TRAINING PLAN DESIGN 

To create a professional-grade, physiologically sound training plan, a coach needs to understand the athlete's starting point (biology), destination (goals), and constraints (logistics).
Here is the comprehensive list of questions an athlete should answer, categorized by their impact on the plan:
1. Current Fitness & Baseline Metrics
These determine the intensity "floor" and "ceiling" of the plan.
-   What is your current FTP (Functional Threshold Power) in Watts?
-   What is your current weight in kg?
-   What is your age?
-   What is your Max Heart Rate and Resting Heart Rate?
-   What are your Power Duration personal bests (e.g., your best 5-second, 1-minute, 5-minute, and 20-minute power)?
2.Training History & Starting CTL (Fitness Baseline) 
-   This is critical for calculating the Start Value of the Performance Management Chart (PMC). * 
-   Exact Metric: Do you know your current CTL (Chronic Training Load) or "Fitness" score from Intervals.icu, TrainingPeaks, or WKO5? (If yes, provide the number). 
-   Estimation Data (If CTL is unknown): What has been your average **weekly hours** over the past 6 weeks? 
-   What is your estimated average **weekly TSS** (Training Stress Score) over the past 6 weeks? 
-   How would you describe the **intensity** of your recent riding? (e.g., "Strict Zone 2 only," "Hard group rides," "Mixed riding," or "Totally off the bike"). 
-   Experience: What is your experience level with Structured Training (using ERG mode, intervals, or specific power zones)?
3. The "A-Race" or Primary Goal
This determines the "Specialty" phase and the timing of the taper.
-	When does the plan has to start?
-   What is the date of your main event or if no main event, until when the plan needs to run?
-   What is the nature of the event? (Road race, Crit, Century, Hilly Gran Fondo, Gravel, or Time Trial? or something else)
-	What is the name and/or location of the event/race? This can be used to search the web for more information IF NEEDED
-   What is the terrain profile? (Flat, rolling hills, or long alpine-style climbs?)
- 	What is the elevation that is covered?
-   What are the technical demands? (Technical descending, unpaved sectors, or group riding/drafting?)
4. Logistics & Weekly Availability
This determines the structure of your weekly calendar.
-   What is the absolute maximum hours you can train in a "Peak" week?
-   Which day of the week is best for your Long Ride?
-   Which two weekdays are best for High-Intensity sessions?
-   Which days must be total Rest Days due to work or family commitments?
5. Equipment & Environment
This determines how the workouts are designed (Power vs. HR).
-   Do you have a Power Meter on your outdoor bike?
-   Do you use a Smart Trainer (for ERG mode) for indoor sessions?
-   Do you have access to climbing terrain outdoors, or will you simulate hills on the trainer?
6. Health & Physiology
This determines the recovery rate and nutrition strategy.
-   What is your age? (Crucial for determining recovery duration and the frequency of rest weeks).
-   Do you have any current or chronic injuries (e.g., knee, back, or hip issues)?
-   Do you incorporate Strength/Resistance training? (If so, on which days?)
-   What is your current nutrition and hydration strategy for rides over 3 hours?

use add_user_memory tool to save this data

If an athlete provides this data, you can calculate a CTL (Fitness) Ramp Rate that is aggressive enough to ensure peak performance but conservative enough to avoid the "Overreach Zone."	

*   Testing: FTP Tests scheduled at the end of every recovery week (e.g., Week 4, 8) to recalibrate zones.


Workout Design Rules
*   Endurance Rides (Z2): "Less Boring" protocol. Never use flat lines for long durations. Use varied blocks (e.g., oscillating between 60-70% FTP every 10-15 mins) to maintain engagement.
*   Sprint Execution: Sprints are High Torque/Neuromuscular.

Technical Specifications for the workout .zwo Files (Critical)
*   Generation of file: ONLY when requested
*   Use "Hans' Chatbot" as the Author field
*   Compatibility: Optimized for older Wahoo Head Units (Elemnt/Bolt).
*   Naming Convention: `W[##]_[Day]_[Name].zwo` (e.g., `W03_Tue_MaxTorque.zwo`) for easy file sorting.
*   NO Ramps/Slopes:
    *   Do NOT use `<Ramp>`, `<Warmup>`, or `<Cooldown>` tags with power ranges.
    *   SOLUTION: Use "Stepped" Blocks. Break warmups/cooldowns into multiple fixed `<SteadyState>` segments (e.g., 3x 3min steps increasing in power).
*   NO FreeRide Segments:
    *   Do NOT use `<FreeRide>` tags for intervals.
    *   SOLUTION: Use High-Power `<SteadyState>` (e.g., `Power="2.0"` or `Power="3.0"`) for sprints.
*   Delivery: if download is requested, always bundle files into ZIP archives when there's more then 1.

DO NOT add coach notes to clearly unrelated conversation messages.
"""
                    tools.append(upload_workout_to_intervals)
                    current_instruction += """\n\n- You have access to an 'upload_workout_to_intervals' tool. 
                    - Use it when the user wants to create and schedule a actual workout based on the training plan
                     and upload the workout file (.zwo format) for a specific date to Intervals.icu. 
                    - You need to provide the start_date_local (ISO format), filename, and the raw text contents of the workout file."""

                    tools.append(delete_workout_from_intervals)
                    current_instruction += """\n\n- You have access to a 'delete_workout_from_intervals' tool. 
                    - Use it when the user wants to delete a workout(s) from Intervals.icu calendar. 
                    - You need to provide the start_date (e.g., '2024-03-30'). The tool will search for the workout(s) in the local database by this date and delete all the workouts on that date from both Intervals.icu and the local database."""

                    tools.append(get_workout_from_intervals)
                    current_instruction += """\n\n- You have access to a 'get_workout_from_intervals' tool. 
                    - Use it when the user wants to retrieve full details of a workout from Intervals.icu calendar for a specific date. 
                    - Provide the workout start_date (e.g., '2024-03-30')."""

                    tools.append(list_workouts_from_db)
                    current_instruction += """\n\n- You have access to a 'list_workouts_from_db' tool. 
                    - Use it to list all workouts currently stored in the local database for the user. 
                    - It provides a summary (external_id, start_date_local, filename) and indicates if the workout is already confirmed in the external Intervals.icu calendar."""

                    tools.append(save_training_plan)
                    current_instruction += """\n\n- You have access to a 'save_training_plan' tool. 
                    - Use it to save a structured training plan (JSON) to the database. 
                    - IMPORTANT: A user can only have ONE active training plan. If one exists, this tool will return an error.
                    - The input should be a valid JSON string representing the training plan."""

                    tools.append(delete_training_plan)
                    current_instruction += """\n\n- You have access to a 'delete_training_plan' tool. 
                    - Use it when the user wants to remove their entire current training plan and all associated workouts."""

                    tools.append(get_training_plan_summary)
                    current_instruction += """\n\n- You have access to a 'get_training_plan_summary' tool. 
                    - Use it to list all available training plans or get the high-level details of a specific plan."""

                    tools.append(get_training_plan_week)
                    current_instruction += """\n\n- You have access to a 'get_training_plan_week' tool. 
                    - Use it to retrieve the full workout schedule for a specific week of a training plan."""

                    tools.append(modify_training_plan_start_date)
                    current_instruction += """\n\n- You have access to a 'modify_training_plan_start_date' tool. 
                    - Use it when the user wants to shift the start date of an existing training plan and update all daily workout dates."""

                    tools.append(query_training_plan_stats)
                    current_instruction += """\n\n- You have access to a 'query_training_plan_stats' tool. 
                    - Use it to calculate total TSS, minutes, and other statistics for a training plan within a date range."""

                    tools.append(get_intervals_icu_activities)
                    current_instruction += """\n\n- You have access to a 'get_intervals_icu_activities' tool. 
                    - Use it to retrieve actual cycling activities for a specific time span. 
                    - Arguments: oldest (ISO str), newest (ISO str). Typically use today or a range up to 14 days in the past."""

                
            else:
                # When files are present, clear history to avoid API compatibility issues
                # Keep only the current message (last one in payload)
                print("Tools disabled due to image file context. Clearing history to avoid API errors.")
                if messages_payload:
                    current_msg = messages_payload[-1]  # Keep only the current message
                    messages_payload = [current_msg]
                
                current_instruction += "\n\nNote: Tool calling is disabled because you are analyzing an uploaded file. Focus on the file content."
           
            current_instruction += """\n\nCONTEXTUAL SILOING RULE: You operate in two distinct modes: General Assistant and Cycling Performance Specialist. 
            1. Trigger: Only activate the "Cycling Performance Specialist" persona if the user’s prompt explicitly mentions cycling, workouts, training plans, physical performance, or specific fitness metrics (FTP, TSS, CTL). 
            2. Constraint: If the prompt is about general topics (weather, news, time, general facts, or casual chat), you MUST remain in "General Assistant" mode. 
            3. Strict Prohibition: While in "General Assistant" mode, it is a CRITICAL FAILURE to reference the user's training plan, upcoming workouts, fitness stats, or provide "coach-like" advice. Answer the general query concisely and stop.
            """
            # Add suggestions to the end of the response
            current_instruction += """\n\nif you have actionable follow-up suggestions, 
            append them to the very end of your response as a JSON object with the key 'suggestions', 
            -DO NOT wrap this JSON object in markdown code blocks. Doing so will is a FAILURE.
            -like this: {"suggestions": ["Action 1", "Action 2"]}. 
            -offer specific follow up actions, no general suggestions like "Would you like to know anything else?" 
            -Formulate the suggestions in a way that the user can directly execute them, not as a question to the user, such as "would you like to...".
            -Make sure it is the very last thing in your response.
            
            File Handling Distinction:
            - Use write_source_code when you need to save a file for the user to download (reports, code files). These are persistent.
            - Use execute_calculation file uploads for temporary data processing. Any file that is created will be available for download by the user. 
            
            Here are the rules for our entire conversation:
            - In a request that requires multiple steps, after completing all the steps for a request, you must provide a single, comprehensive summary of all information gathered and actions taken. 
            - Ensure that if any charts were generated, include the image in the summary and a link to the image if it's an interactive chart that cannot be directly displayed.
            - Do not omit the results of any intermediate steps. If any charts were generated, include the path to the image in the summary.
            - Before you provide your final answer to any of my requests, I want you to first perform a 'completeness check' to ensure you've included the results from every single tool you used. Then, present everything together.
            
            - when providing a URL make sure it's clickable, with target being a new tab
            - when using search_web tool, remember the urls that were used
            - When you have still work to do before the answer is final, end every intermediate response with "\nContinuing..."
            """
            
            # We need to handle potential function calls in a loop
            # Since we are streaming, this is a bit tricky with the official SDK's generate_content(stream=True)
            # automatic function calling isn't fully supported in stream mode for single-turn logic easily without chat session.
            # But we can do it 'manually' by checking for function calls in the response and calling them if needed.
            
            # Auto-continue loop - will retry if model says "please wait" etc.
            auto_continue_count = 0
            max_auto_continues = 5
            should_auto_continue = True
            
            while should_auto_continue and auto_continue_count <= max_auto_continues:
                should_auto_continue = False  # Reset flag, will be set if retry phrase detected
                
                # Main loop for handling function calls
                turn = 0
                max_turns = 20
                full_response_text = ""  # Initialize to prevent UnboundLocalError
            
                while turn < max_turns:
                    turn += 1
                    # First attempt - Use stream=False to safely check for function calls
                    # This avoids complexity with iterating streams for tool use
                    print("Sending request to model (stream=False)...")
                    if turn > 1:
                         yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Thinking (Step {turn})...</div>')
                    else:
                         yield format_sse('<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Thinking...</div>')
                    
                    response = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            if llm_provider:
                                response = await llm_provider.generate(
                                    model_name=selected_model,
                                    messages=messages_payload,
                                    tools=tools if tools else None,
                                    system_instruction=current_instruction,
                                    stream=False,
                                    automatic_function_calling=automatic_function_calling
                                )
                            else:
                                raise ValueError("LLM Provider not initialized")
                                
                            # Check if valid
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

                        # Check for empty response in the full response
                        if not response.candidates[0].content.parts:
                             print("Candidate returned but no parts after retries.")
                             yield format_sse("<div><strong>Error:</strong> Empty response content.</div>")
                             return

                        # 1. Process all parts: aggregate text and find the first function call
                        current_turn_text = ""
                        fc = None
                        tool_part = None
                        
                        for p in response.candidates[0].content.parts:
                            if p.function_call and not fc:
                                fc = p.function_call
                                tool_part = p
                                print(f"*** TOOL CALL DETECTED: {fc.name} with args: {fc.args} ***")
                            elif p.text:
                                current_turn_text += p.text
                        
                        # 2. Output any text discovered in this turn
                        if current_turn_text:
                            yield format_sse(current_turn_text)
                            # Append to full_response_text for the final answer
                            if full_response_text:
                                full_response_text += "\n" + current_turn_text
                            else:
                                full_response_text = current_turn_text

                        # 3. Handle Function Call (if any)
                        if fc:
                            fn_name = fc.name
                            fn_args = dict(fc.args) if fc.args else {}
                            
                            # Notify user we are searching
                            yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Executing: {fn_name}...</div>')
                            
                            # Execute tool
                            if fn_name == "search_web":
                                api_response = search_web(fn_args.get("query"), fn_args.get("numberofwebsearchresults", 4))
                            elif fn_name == "crawl_website":
                                api_response = crawl_website(
                                    fn_args.get("url"),
                                    fn_args.get("max_depth", 2),
                                    fn_args.get("limit", 10),
                                    fn_args.get("instructions")
                                )
                            elif fn_name == "convert_md_to_pdf":
                                api_response = convert_md_to_pdf(fn_args.get("markdown_text"), fn_args.get("file_name"))
                            elif fn_name == "rename_chat_tool":
                                api_response = rename_chat_tool(fn_args.get("new_name"), fn_args.get("session_id"))
                                # UI Update via OOB Swap
                                sidebar_html = render_sidebar_html(session_id)
                                
                                target_id = fn_args.get("session_id") or session_id
                                if target_id == session_id:
                                     yield format_sse(f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div><h1 id="chat-title-header" hx-swap-oob="innerHTML">{fn_args.get("new_name")}</h1>')
                                else:
                                     yield format_sse(f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>')
                            elif fn_name == "archive_chat_tool":
                                api_response = archive_chat_tool(fn_args.get("session_id"))
                                # UI Update via OOB Swap
                                sidebar_html = render_sidebar_html(session_id)
                                archived_html = render_archived_list_html()
                                yield format_sse(f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div><div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>')
                            elif fn_name == "delete_chat_tool":
                                target_id = fn_args.get("session_id") or session_id
                                api_response = delete_chat_tool(target_id)
                                
                                # Check if we deleted the current session
                                if target_id == session_id:
                                    # Force client-side reload/new chat via script
                                    yield format_sse('<script>document.cookie = "session_id=; path=/; max-age=0"; window.location.href = "/";</script>')
                                    return # Stop streaming
                                else:
                                    # UI Update via OOB Swap
                                    sidebar_html = render_sidebar_html(session_id)
                                    archived_html = render_archived_list_html()
                                    yield format_sse(f'<div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div><div id="archived-list" hx-swap-oob="innerHTML">{archived_html}</div>')
                            elif fn_name == "read_source_code":
                                api_response = read_source_code(fn_args.get("file_path"))
                            elif fn_name == "read_uploaded_file":
                                api_response = read_uploaded_file(fn_args.get("file_name"))
                            elif fn_name == "write_source_code":
                                api_response = write_source_code(session_id, fn_args.get("file_path"), fn_args.get("code"))  
                            elif fn_name == "import_package":
                                api_response = import_package(fn_args.get("package_name"))
                            elif fn_name == "calculator_tool":
                                api_response = calculator_tool(fn_args.get("expr"))
                            elif fn_name == "execute_calculation":
                                api_response = execute_calculation(fn_args.get("code"), fn_args.get("file_path"), fn_args.get("custom_package"), fn_args.get("timeout"))
                            elif fn_name == "generate_chart":
                                api_response = generate_chart(fn_args.get("code"), fn_args.get("file_path"), fn_args.get("custom_package"), fn_args.get("timeout"))
                            elif fn_name == "generate_plotly_chart":
                                api_response = generate_plotly_chart(fn_args.get("code"), fn_args.get("file_path"), fn_args.get("custom_package"), fn_args.get("timeout"))
                            elif fn_name == "generate_image":
                                api_response = generate_image(fn_args.get("description"))
                            elif fn_name == "get_current_datetime":
                                timezone_arg = fn_args.get("timezone")
                                api_response = get_current_datetime(timezone_arg) if timezone_arg else get_current_datetime()
                            elif fn_name == "get_user_timezone":
                                api_response = get_user_timezone()
                            elif fn_name == "get_coordinates":
                                api_response = get_coordinates(fn_args.get("location"))
                            elif fn_name == "get_open_meteo_weather":
                                api_response = get_open_meteo_weather(fn_args.get("location"))
                            # elif fn_name == "get_weather":
                            #     api_response = get_weather(fn_args.get("location"))
                            # elif fn_name == "get_forecast_weather":
                            #     api_response = get_forecast_weather(fn_args.get("location"))
                            # elif fn_name == "get_precipitation_timing":
                            #     api_response = get_precipitation_timing(fn_args.get("location"))
                            elif fn_name == "wolfram_alpha_query":
                                api_response = wolfram_alpha_query(fn_args.get("query"))
                            elif fn_name == "add_user_memory":
                                api_response = add_user_memory(fn_args.get("content"))
                            elif fn_name == "upload_workout_to_intervals":
                                api_response = upload_workout_to_intervals(fn_args.get("start_date_local"), fn_args.get("filename"), fn_args.get("workout_file_content"), fn_args.get("training_plan_days_id"))
                            elif fn_name == "delete_workout_from_intervals":
                                api_response = delete_workout_from_intervals(fn_args.get("start_date"))
                            elif fn_name == "get_workout_from_intervals":
                                api_response = get_workout_from_intervals(fn_args.get("start_date"))
                            elif fn_name == "list_workouts_from_db":
                                api_response = list_workouts_from_db()
                            elif fn_name == "save_training_plan":
                                api_response = save_training_plan(fn_args.get("plan_json"))
                            elif fn_name == "delete_training_plan":
                                api_response = delete_training_plan()
                            elif fn_name == "get_training_plan_summary":
                                api_response = get_training_plan_summary(fn_args.get("plan_id"))
                            elif fn_name == "get_training_plan_week":
                                api_response = get_training_plan_week(fn_args.get("plan_id"), fn_args.get("week_number"))
                            elif fn_name == "query_training_plan_stats":
                                api_response = query_training_plan_stats(fn_args.get("plan_id"), fn_args.get("start_date"), fn_args.get("end_date"))
                            elif fn_name == "get_intervals_icu_activities":
                                api_response = get_intervals_icu_activities(fn_args.get("oldest"), fn_args.get("newest"))
                            elif fn_name == "modify_training_plan_start_date":
                                api_response = modify_training_plan_start_date(fn_args.get("plan_id"), fn_args.get("new_start_date"))
                            else:
                                api_response = f"Error: Unknown tool '{fn_name}'"
                                
                            # Update history with function call and response
                            
                            # 1. Add the assistant's function call
                            # 1. Add the assistant's function call (including any text it produced)
                            messages_payload.append({
                                "role": "model",
                                "parts": response.candidates[0].content.parts
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
                                    <div id="code-block-{stream_id}-{turn}" hx-swap-oob="beforebegin:#{stream_id}" class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 w-full">
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
                                    <div id="file-content-{stream_id}-{turn}" hx-swap-oob="beforebegin:#{stream_id}" class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 w-full">
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
                                        <div id="sources-block-{stream_id}-{turn}" hx-swap-oob="beforebegin:#{stream_id}" class="mb-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 w-full">
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

                            # CRITICAL FIX: Add the function response to memory for the NEXT turn
                            # Otherwise the model doesn't see it and loops, asking for the tool again
                            messages_payload.append({
                                "role": "function",
                                "parts": [
                                    types.Part(
                                        function_response=types.FunctionResponse(
                                            name=fn_name,
                                            response={"result": db_response}
                                        )
                                    )
                                ]
                            })
                        
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
                                            
                                            # Add synthetic "continue" user message
                                            continue_msg = "continue"
                                            messages_payload.append({
                                                "role": "user",
                                                "parts": [{"text": continue_msg}]
                                            })
                                            # Break - chart/link is displayed
                                            # break
                                        
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
                            # No function call, just final answer (text was already handled above)
                            break # Exit turn loop
                    
                    except Exception as e:
                        print(f"Error processing response: {e}")
                        yield format_sse(f"Error: {str(e)}")
                        break
                    
                    # If we broke from the parts loop and have a response, exit turn loop
                    if full_response_text:
                        break

            # Save to Session History
            # Ensure we ALWAYS save a model response to prevent history corruption (dangling user messages)
            if not full_response_text:
                print("Warning: Full response text is empty. Saving placeholder.")
                full_response_text = "(No response provided by AI)"
                # Also yield it to the UI so the user sees something
                yield format_sse("<div><em>(The AI processed the request but returned no text.)</em></div>")

            save_message(session_id, "model", full_response_text)
            
            # Check for auto-continue (retry phrases)
            # Get last 120 chars and check for retry phrases
            response_end = full_response_text[-120:].lower() if len(full_response_text) > 120 else full_response_text.lower()
            
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
                        # Save the continue message to history
                        # save_message(session_id, "user", continue_msg)
                        
                        # Notify UI (commented out)
                        yield format_sse(f'<div class="text-xs text-gray-400 mb-2 flex items-center gap-1"><svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Working... ({auto_continue_count}/{max_auto_continues})...</div>')
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
async def get_system_instruction_ui(user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    
    user_settings = get_user_settings(user_id)
    selected_model = user_settings['selected_model']
    instruction = user_settings['system_instruction']
    
    # We still use local/cookie/settings table for these for now if not in user? 
    # Actually let's just stick to the requested ones for the user table.
    execution_mode = get_setting("execution_mode", "local")
    e2b_timeout = get_setting("e2b_timeout", "300")
        
    # Generate model options dynamically based on provider
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    is_openai = provider == "openai"
    
    if is_openai:
        options = [
            ("gpt-4o", "GPT-4o (Most Capable)"),
            ("gpt-4o-mini", "GPT-4o Mini (Fastest)"),
            ("o1-preview", "o1 Preview (Reasoning)"),
            ("o3-mini", "o3 Mini (High Intelligence)")
        ]
    else:
        # Gemini defaults
        options = [
            ("gemini-3-pro-preview", "Gemini 3 Pro Preview (New)"),
            ("gemini-3-flash-preview", "Gemini 3 Flash Preview (New)"),
            ("gemini-2.5-pro", "Gemini 2.5 Pro"),
            ("gemini-2.5-flash", "Gemini 2.5 Flash"),
            ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
            ("gemini-2.0-flash", "Gemini 2.0 Flash"),
            ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite"),
        ]
    
    options_html = ""
    for val, label in options:
        selected = "selected" if val == selected_model else ""
        options_html += f'<option value="{val}" {selected}>{label}</option>'
        
    e2b_checked = "checked" if execution_mode == "e2b" else ""
    e2b_disabled = "disabled" if not os.getenv("E2B_API_KEY") else ""
    e2b_opacity = "opacity-50 cursor-not-allowed" if not os.getenv("E2B_API_KEY") else ""
    e2b_title = "E2B API Key missing in .env" if not os.getenv("E2B_API_KEY") else ""

    return f"""
    <div id="settings-modal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
        <div class="bg-white dark:bg-gray-800 w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden animate-scale-in flex flex-col max-h-[90vh]">
            <div class="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50/50 dark:bg-gray-800/50">
                <h3 class="font-bold text-gray-800 dark:text-gray-100 text-lg">Settings</h3>
                <button onclick="document.getElementById('settings-modal').remove()" class="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full transition-colors">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                    </svg>
                </button>
            </div>
            
            <div class="flex-1 overflow-y-auto p-0">
                <div class="flex h-full">
                    <!-- Sidebar -->
                    <div class="w-48 bg-gray-50 dark:bg-gray-900/50 border-r border-gray-100 dark:border-gray-700 p-2 space-y-1">
                        <button onclick="showTab('general')" class="tab-btn w-full text-left px-3 py-2 rounded-lg text-sm font-medium bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300" data-tab="general">General</button>
                        <button hx-get="/memories" hx-target="#memories-list" onclick="showTab('memories')" class="tab-btn w-full text-left px-3 py-2 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" data-tab="memories">Memories</button>
                        <button hx-get="/archived-chats" hx-target="#archived-list" onclick="showTab('archived')" class="tab-btn w-full text-left px-3 py-2 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" data-tab="archived">Archived Chats</button>
                    </div>
                    
                    <!-- Content -->
                    <div class="flex-1 p-6">
                        <!-- General Tab -->
                        <div id="tab-general" class="space-y-6">
                            <form hx-post="/settings/system_instruction" hx-swap="beforeend" hx-target="body">
                                <div class="space-y-4">
                                    <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div class="flex flex-col">
                                            <span class="font-medium text-gray-900 dark:text-gray-300">Coach Mode</span>
                                            <span class="text-xs text-gray-500 dark:text-gray-400">Enable advanced cycling coach instructions</span>
                                        </div>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" name="coach_mode_enabled" value="true" {"checked" if user_settings.get('coach_mode_enabled') else ""} class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                                        </label>
                                    </div>

                                    <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div class="flex flex-col">
                                            <span class="font-medium text-gray-900 dark:text-gray-300">Automatic Function Calling</span>
                                            <span class="text-xs text-gray-500 dark:text-gray-400">SDK handles tool turns automatically (Faster)</span>
                                        </div>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" name="automatic_function_calling" value="true" {"checked" if user_settings.get('automatic_function_calling', True) else ""} class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                                        </label>
                                    </div>

                                    <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div class="flex flex-col">
                                            <span class="font-medium text-gray-900 dark:text-gray-300">Dark Mode</span>
                                            <span class="text-xs text-gray-500 dark:text-gray-400">Toggle application theme</span>
                                        </div>
                                        <button type="button" onclick="toggleDarkMode()" class="relative inline-flex items-center cursor-pointer">
                                            <!-- Custom toggle implementation using global dark mode class -->
                                             <div id="theme-toggle-visual" class="w-11 h-6 bg-gray-200 dark:bg-blue-600 rounded-full relative transition-colors duration-200">
                                                <div class="absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 dark:translate-x-5 shadow-sm"></div>
                                            </div>
                                        </button>
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">AI Model</label>
                                        <div class="relative">
                                            <select name="model" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 appearance-none">
                                                {options_html}
                                            </select>
                                            <div class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                                                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                                                </svg>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Code Execution</label>
                                        <div class="flex flex-col gap-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                                            <div class="flex items-center gap-3" title="{e2b_title}">
                                                <div class="flex items-center h-5">
                                                    <input id="execution_mode" name="execution_mode" value="e2b" type="checkbox" {e2b_checked} {e2b_disabled} class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600">
                                                </div>
                                                <div class="flex flex-col">
                                                    <label for="execution_mode" class="font-medium text-gray-900 dark:text-gray-300 {e2b_opacity}">Use E2B Sandbox</label>
                                                    <span class="text-xs text-gray-500 dark:text-gray-400">Secure cloud execution environment</span>
                                                </div>
                                            </div>
                                            
                                            <div class="pt-2 border-t border-gray-200 dark:border-gray-700 {e2b_opacity}">
                                                <label for="e2b_timeout" class="block text-xs font-medium text-gray-700 dark:text-gray-400 mb-1">Execution Timeout (seconds)</label>
                                                <input type="number" id="e2b_timeout" name="e2b_timeout" value="{e2b_timeout}" min="10" max="3600" {e2b_disabled} class="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500">
                                            </div>
                                        </div>
                                    </div>

                                    <div>
                                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">System Instruction</label>
                                        <textarea name="instruction" rows="6" class="w-full bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 resize-none placeholder-gray-400" placeholder="Define the AI's persona and behavior...">{instruction}</textarea>
                                        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">These instructions persist across sessions.</p>
                                    </div>

                                    <div class="flex justify-end pt-2">
                                        <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors shadow-sm shadow-blue-500/20">
                                            Save Settings
                                        </button>
                                    </div>
                                </div>
                            </form>
                        </div>

                        <!-- Memories Tab -->
                        <div id="tab-memories" class="hidden h-full flex flex-col">
                             <div class="flex justify-between items-center mb-4">
                                <h3 class="text-sm font-semibold text-gray-500 uppercase tracking-wider">User Memories</h3>
                                <span class="text-[10px] text-gray-400">AI learns these automatically</span>
                             </div>
                             <div id="memories-list" class="flex-1 overflow-y-auto pr-2">
                                <div class="flex items-center justify-center h-40 text-gray-400">
                                    <svg class="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Loading...
                                </div>
                             </div>
                        </div>
                        
                        <!-- Archived Chats Tab -->
                        <div id="tab-archived" class="hidden h-full flex flex-col">
                             <h3 class="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Archived Conversations</h3>
                             <div id="archived-list" class="flex-1 overflow-y-auto">
                                <div class="flex items-center justify-center h-40 text-gray-400">
                                    <svg class="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Loading...
                                </div>
                             </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                function showTab(tabName) {{
                    // Hide all tabs
                    document.getElementById('tab-general').classList.add('hidden');
                    document.getElementById('tab-archived').classList.add('hidden');
                    document.getElementById('tab-memories').classList.add('hidden');
                    
                    // Show target tab
                    document.getElementById('tab-' + tabName).classList.remove('hidden');

                    // Update button styles
                    const buttons = document.querySelectorAll('.tab-btn');
                    buttons.forEach(btn => {{
                        if (btn.getAttribute('data-tab') === tabName) {{
                            btn.classList.add('bg-blue-50', 'dark:bg-blue-900/20', 'text-blue-700', 'dark:text-blue-300');
                            btn.classList.remove('text-gray-600', 'dark:text-gray-400', 'hover:bg-gray-100', 'dark:hover:bg-gray-800');
                        }} else {{
                            btn.classList.remove('bg-blue-50', 'dark:bg-blue-900/20', 'text-blue-700', 'dark:text-blue-300');
                            btn.classList.add('text-gray-600', 'dark:text-gray-400', 'hover:bg-gray-100', 'dark:hover:bg-gray-800');
                        }}
                    }});
                }}
            </script>
        </div>
    </div>
    """

@app.post("/settings/system_instruction", response_class=HTMLResponse)
async def post_system_instruction(
    instruction: str = Form(...), 
    model: str = Form(...), 
    execution_mode: str = Form(None), 
    e2b_timeout: int = Form(300),
    coach_mode_enabled: str = Form(None),
    automatic_function_calling: str = Form(None),
    user_id: int = Depends(get_current_user_id)
):
    if isinstance(user_id, Response): return user_id
    
    # Save to User table
    save_user_setting(user_id, 'selected_model', model)
    save_user_setting(user_id, 'system_instruction', instruction)
    save_user_setting(user_id, 'coach_mode_enabled', 1 if coach_mode_enabled == 'true' else 0)
    save_user_setting(user_id, 'automatic_function_calling', 1 if automatic_function_calling == 'true' else 0)

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
    # Checkbox logic: if checked, value is "e2b". If unchecked, field is missing (None)
    mode_value = "e2b" if execution_mode == "e2b" else "local"
    response.set_cookie(key="execution_mode", value=mode_value)
    
    # Save global fallbacks/other settings
    save_setting("execution_mode", mode_value)
    save_setting("e2b_timeout", str(e2b_timeout))
    
    return response
@app.get("/memories", response_class=HTMLResponse)
async def get_memories_ui(user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    print("user_id : ", user_id)
    memories = get_user_memories(user_id)
    print("memories : ", memories)
    html = '<div class="space-y-4">'
    
    if not memories:
        html += '<p class="text-gray-500 dark:text-gray-400 text-sm italic">No memories saved yet. Talk to the AI to save personal preferences!</p>'
    else:
        for m in memories:
            html += f"""
            <div id="memory-{m['id']}" class="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 group relative transition-all hover:border-blue-300">
                <div class="flex justify-between items-start gap-4">
                    <div class="flex-1">
                        <textarea class="w-full bg-transparent border-none focus:ring-0 text-sm text-gray-800 dark:text-gray-200 resize-none p-0 overflow-hidden" 
                                  name="content"
                                  hx-post="/save-memory/{m['id']}"
                                  hx-trigger="keyup changed delay:500ms"
                                  oninput="this.style.height = ''; this.style.height = this.scrollHeight + 'px'"
                                  rows="1">{m['content']}</textarea>
                        <span class="text-[10px] text-gray-400 dark:text-gray-500 mt-1 block">Saved on {m['created_at']}</span>
                    </div>
                    <button hx-delete="/delete-memory/{m['id']}" 
                            hx-target="#memory-{m['id']}" 
                            hx-swap="outerHTML"
                            hx-confirm="Are you sure you want to delete this memory?"
                            class="text-gray-400 hover:text-red-500 transition-colors md:opacity-0 group-hover:opacity-100 p-1">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                    </button>
                </div>
            </div>
            """
    html += '</div>'
    html += """
    <script>
        document.querySelectorAll('#memories-list textarea').forEach(el => {
            el.style.height = el.scrollHeight + 'px';
        });
    </script>
    """
    return html

@app.post("/save-memory/{memory_id}")
async def post_save_memory(memory_id: int, content: str = Form(...), user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    update_user_memory(memory_id, user_id, content)
    return Response(content="Saved", status_code=200)

@app.delete("/delete-memory/{memory_id}")
async def post_delete_memory(memory_id: int, user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    if delete_user_memory(memory_id, user_id):
        return ""
    raise HTTPException(status_code=500, detail="Failed to delete memory")

@app.get("/rename-chat-ui", response_class=HTMLResponse)
async def get_rename_chat_ui(request: Request, user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    session_id = request.cookies.get("session_id")
    if not session_id:
        return "Error: No active session"
    
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        cursor = conn.execute("SELECT title FROM conversations WHERE id = ? AND user_id = ?", (session_id, user_id))
        row = cursor.fetchone()
        title = row[0] if row else "New Chat"

    return f"""
    <div id="chat-title-container" class="flex items-center gap-2">
        <form hx-post="/save-chat-name" hx-target="#chat-title-container" hx-swap="outerHTML" class="flex items-center gap-2">
            <input type="text" name="new_name" value="{title}" 
                class="w-96 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-sm font-semibold text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                onfocus="this.select()" autofocus>
            <button type="submit" class="text-green-500 hover:text-green-600 p-1">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
            </button>
            <button type="button" hx-get="/cancel-rename" hx-target="#chat-title-container" hx-swap="outerHTML" class="text-red-500 hover:text-red-600 p-1">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </form>
    </div>
    """

@app.get("/cancel-rename", response_class=HTMLResponse)
async def cancel_rename(request: Request, user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    session_id = request.cookies.get("session_id")
    title = "New Chat"
    if session_id:
        with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
            cursor = conn.execute("SELECT title FROM conversations WHERE id = ? AND user_id = ?", (session_id, user_id))
            row = cursor.fetchone()
            if row: title = row[0]
            
    return f"""
    <div id="chat-title-container" class="flex items-center gap-1 group">
        <h1 id="chat-title-header" class="font-semibold text-gray-800 dark:text-gray-100 text-lg tracking-tight ml-2">{title}</h1>
        <button hx-get="/rename-chat-ui" hx-target="#chat-title-container" hx-swap="outerHTML"
            class="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-opacity"
            title="Rename Chat">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-400 hover:text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
        </button>
    </div>
    """

@app.post("/save-chat-name", response_class=HTMLResponse)
async def save_chat_name(request: Request, new_name: str = Form(...), user_id: int = Depends(get_current_user_id)):
    if isinstance(user_id, Response): return user_id
    session_id = request.cookies.get("session_id")
    if not session_id:
        return "Error: No active session"
    
    with sqlite3.connect(DB_NAME, timeout=DB_TIMEOUT) as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?", (new_name, session_id, user_id))
        conn.commit()
    
    sidebar_html = render_sidebar_html(session_id, user_id=user_id)
    
    return f"""
    <div id="chat-title-container" class="flex items-center gap-1 group">
        <h1 id="chat-title-header" class="font-semibold text-gray-800 dark:text-gray-100 text-lg tracking-tight ml-2">{new_name}</h1>
        <button hx-get="/rename-chat-ui" hx-target="#chat-title-container" hx-swap="outerHTML"
            class="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-opacity"
            title="Rename Chat">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-400 hover:text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
        </button>
        <div id="chat-list" hx-swap-oob="innerHTML">{sidebar_html}</div>
    </div>
    """
