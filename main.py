import os
import asyncio
import uuid
import sqlite3
import markdown
import io
import sys
import traceback
import datetime
from collections import deque
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
except ImportError:
    pass # Matplotlib might not be installed yet, which is fine for basic usage

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
    return "You are a helpful AI assistant."

def save_system_instruction(instruction: str):
    with open(SYSTEM_INSTRUCTION_FILE, "w") as f:
        f.write(instruction)

IMAGE_GEN_INSTRUCTION = """
You can generate artistic or creative images by using the following markdown format:
![Image Description](https://image.pollinations.ai/prompt/{description}?width=1024&height=1024&nologo=true)
IMPORTANT: Use this ONLY for artistic requests (e.g. "draw a cat").
For data visualizations, charts, graphs, or plots, you MUST use the `execute_python` tool with matplotlib.
"""

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- 2. DATABASE & SESSION MANAGEMENT ---
DB_NAME = "chat.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                mime_type TEXT
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

# Initialize DB on startup
init_db()

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieves history for a specific session ID from DB."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT role, content, image_path FROM messages WHERE session_id = ? ORDER BY id ASC", 
            (session_id,)
        )
        rows = cursor.fetchall()
        # Convert to format expected by Gemini and our logic
        history = []
        for row in rows:
            content = row["content"]
            role = row["role"]
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

            # Note: We are currently NOT loading historical images into the context for Gemini to save tokens/bandwidth
            # and because multi-turn image chat can be complex. 
            # If needed, we could load the image here using PIL.
            history.append({"role": role, "parts": parts})
        return history

def save_message(session_id: str, role: str, content: str, image_path: Optional[str] = None, mime_type: Optional[str] = None):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, image_path, mime_type) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, image_path, mime_type)
        )

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

def render_bot_message(content: str, stream_id: Optional[str] = None, final: bool = False) -> str:
    # If it's a stream placeholder
    if stream_id and not final:
        return f"""
        <div id="{stream_id}" 
             hx-ext="sse" 
             sse-connect="/stream?prompt={urllib.parse.quote(content)}&stream_id={stream_id}" 
             sse-swap="message" 
             class="flex justify-start mb-4 animate-fade-in">
            <div class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
                <span id="cursor" class="inline-block w-2 h-5 bg-blue-500 cursor-blink align-middle"></span>
            </div>
        </div>
        """
    
    # If it's the final content (or historical content)
    # Enable tables extension
    html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
    
    # If it's an OOB swap update
    if stream_id and final:
        return f"""
        <div id="{stream_id}" 
             hx-swap-oob="outerHTML:#{stream_id}" 
             class="flex justify-start mb-4">
            <div class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
                {html_content}
            </div>
        </div>
        """
    
    # Just static HTML (for history)
    return f"""
    <div class="flex justify-start mb-4">
        <div class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
            {html_content}
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

# --- 3. TOOLS ---
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
        # Format results for the model
        context = "\n".join([
            f"Source: {result['url']}\nContent: {result['content']}" 
            for result in response.get('results', [])[:3] # Limit to top 3
        ])
        return context
    except Exception as e:
        return f"Error performing search: {str(e)}"

def execute_python(code: str):
    """
    Executes the given Python code and returns the standard output.
    Use this tool for calculations, data processing, or logic that requires code execution.
    The code must print the final result to stdout.
    If the code generates an image (e.g. using matplotlib), save it to the current directory.
    """
    print(f"Executing Python code:\n{code}")
    
    # Create a string buffer to capture stdout
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    
    # Ensure generated directory exists
    generated_dir = "static/generated"
    os.makedirs(generated_dir, exist_ok=True)
    
    # Snapshot current directory files to detect new ones
    # We only look for images in the current working directory (root)
    cwd_files_before = set(os.listdir('.'))
    
    try:
        # Execute the code
        exec_globals = {}
        exec(code, exec_globals)
        
        # Get the output
        output = redirected_output.getvalue()
        if not output:
            output = "Code executed successfully (no output)."
            
        # Check for new files
        cwd_files_after = set(os.listdir('.'))
        new_files = cwd_files_after - cwd_files_before
        
        generated_images = []
        for filename in new_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                # Move to static/generated
                src = filename
                dst = f"{generated_dir}/{uuid.uuid4()}_{filename}"
                os.rename(src, dst)
                generated_images.append(f"/{dst}")
                print(f"Captured generated image: {dst}")
        
        # Append image links to output
        if generated_images:
            output += "\n\n### Generated Images:\n"
            for img_path in generated_images:
                output += f"![Generated Image]({img_path})\n"
                
        return output
    except Exception:
        # Capture the traceback if an error occurs
        return f"Error executing code:\n{traceback.format_exc()}"
    finally:
        # Restore stdout
        sys.stdout = old_stdout

def get_current_datetime():
    """
    Returns the current date and time.
    Use this tool when the user asks about the current time, date, or day of the week.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        
    # Load history to render
    history_items = get_history(session_id)
    chat_history_html = ""
    
    # We need to fetch image_path for rendering history, so get_history should probably return it.
    # But get_history returns formatted parts. Let's adjust get_history slightly or just query manually here?
    # Actually get_history above was updated to return parts only.
    # Let's update get_history to return the raw row data or handle rendering there?
    # Better: Let's just query DB here for rendering to keep get_history clean for Gemini.
    
    with sqlite3.connect(DB_NAME) as conn:
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
        
        if role == "user":
            chat_history_html += render_user_message(content, image_path)
        else:
            chat_history_html += render_bot_message(content)

    resp = templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history_html})
    # Set the cookie on the browser so it remembers us
    resp.set_cookie(key="session_id", value=session_id)
    return resp

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
async def stream_response(prompt: str, session_id: str = Cookie(None), stream_id: str = Query(...), image_path: Optional[str] = None, mime_type: Optional[str] = None):
    """
    The 'session_id' is automatically extracted from the browser cookie.
    """
    async def event_generator():
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
                    
                except Exception as e:
                    print(f"Error uploading file: {e}")
                    yield format_sse(f"<div><strong>Error:</strong> Failed to upload file to AI: {str(e)}</div>")
                    # Continue without file? Or return? Let's return to be safe.
                    return

            messages_payload.append({
                "role": "user",
                "parts": current_parts
            })

            # 2. Call Gemini
            # Instantiate model with the current system instruction
            current_instruction = get_system_instruction()
            # Fallback to a default if empty, or just pass None (Gemini handles None fine usually, but let's be safe)
            if not current_instruction:
                current_instruction = "You are a helpful AI assistant."
            
            # Append Image Generation Instruction
            current_instruction += f"\n\n{IMAGE_GEN_INSTRUCTION}"
            
            # Define tools
            tools = []
            if tavily_client:
                tools.append(search_web)
                current_instruction += "\n\nYou have access to a 'search_web' tool. Use it whenever the user asks for current information, news, or facts you don't know. Do NOT invent new tools. Only use 'search_web' for searching."
            
            tools.append(execute_python)
            current_instruction += "\n\nYou have access to an 'execute_python' tool. Use it for calculations, math, or complex logic. It is allowed to import any Python module (including matplotlib). The code MUST print the result to stdout. If you generate plots or images, save them to the current directory (e.g. 'plot.png'). The tool will capture them and provide a markdown link (e.g. `![Generated Image](/static/generated/...)`). You MUST include this markdown link in your final response to the user."

            tools.append(get_current_datetime)
            current_instruction += "\n\nYou have access to a 'get_current_datetime' tool. Use it when asked about the current time or date."

            tools.append(get_coordinates)
            current_instruction += "\n\nYou have access to a 'get_coordinates' tool. Use it to find the latitude and longitude of locations."

            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=current_instruction,
                tools=tools
            )
            
            # We need to handle potential function calls in a loop
            # Since we are streaming, this is a bit tricky with the official SDK's generate_content(stream=True)
            # automatic function calling isn't fully supported in stream mode for single-turn logic easily without chat session.
            # But we can do it manually.
            
            # Multi-turn loop
            max_turns = 5
            turn = 0
            
            while turn < max_turns:
                turn += 1
                print(f"Turn {turn}/{max_turns}...")
                
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
                        # Check for 429 in string representation just in case
                        if "429" in str(e) or "quota" in str(e).lower():
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
                        elif fn_name == "execute_python":
                            api_response = execute_python(fn_args.get("code"))
                        elif fn_name == "get_current_datetime":
                            api_response = get_current_datetime()
                        elif fn_name == "get_coordinates":
                            api_response = get_coordinates(fn_args.get("location"))
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
                        
                        # 2. Add the function response
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
                        fr_json = json.dumps({
                            "function_response": {
                                "name": fn_name,
                                "response": {"result": api_response}
                            }
                        })
                        save_message(session_id, "function", fr_json)
                        
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

            # 3. Save to Session History
            # Ensure we ALWAYS save a model response to prevent history corruption (dangling user messages)
            if not full_response_text:
                print("Warning: Full response text is empty. Saving placeholder.")
                full_response_text = "(No response provided by AI)"
                # Also yield it to the UI so the user sees something
                yield format_sse("<div><em>(The AI processed the request but returned no text.)</em></div>")

            save_message(session_id, "model", full_response_text)

            # 4. Finalize UI (OOB Swap)
            final_div = render_bot_message(full_response_text, stream_id=stream_id, final=True)
            
            yield format_sse(final_div)
            yield format_sse("", event="close")

        except Exception as e:
            yield format_sse(error_div)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/reset")
async def reset_chat(session_id: str = Cookie(None)):
    if session_id:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    return ""

@app.get("/settings/system_instruction", response_class=HTMLResponse)
async def get_system_instruction_ui():
    instruction = get_system_instruction()
    return f"""
    <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in" id="settings-modal">
        <div class="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden transform transition-all">
            <div class="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                <h3 class="text-lg font-semibold text-gray-800">System Settings</h3>
                <button onclick="document.getElementById('settings-modal').remove()" class="text-gray-400 hover:text-gray-600 transition-colors">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            <div class="p-6 space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">System Instruction (Persona)</label>
                    <textarea name="instruction" 
                              class="w-full h-32 p-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-sm resize-none"
                              placeholder="e.g., You are a helpful assistant...">{instruction}</textarea>
                    <p class="text-xs text-gray-500 mt-2">This instruction will be applied to all new messages.</p>
                </div>
            </div>
            <div class="p-6 border-t border-gray-100 bg-gray-50 flex justify-end gap-3">
                <button onclick="document.getElementById('settings-modal').remove()" 
                        class="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors">
                    Cancel
                </button>
                <button hx-post="/settings/system_instruction" 
                        hx-include="[name='instruction']"
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
async def post_system_instruction(instruction: str = Form(...)):
    save_system_instruction(instruction)
    # Return a success notification or close the modal
    return """
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
    """