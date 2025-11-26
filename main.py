import os
import asyncio
import uuid
import markdown
from collections import deque
from typing import Dict
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, Response, Cookie
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai

# --- 1. CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Remove global model instantiation to allow dynamic system instructions
# model = genai.GenerativeModel('gemini-2.5-flash')

SYSTEM_INSTRUCTION_FILE = "system_instruction.txt"

def get_system_instruction() -> str:
    if os.path.exists(SYSTEM_INSTRUCTION_FILE):
        with open(SYSTEM_INSTRUCTION_FILE, "r") as f:
            return f.read().strip()
    return "You are a helpful AI assistant."

def save_system_instruction(instruction: str):
    with open(SYSTEM_INSTRUCTION_FILE, "w") as f:
        f.write(instruction)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- 2. SESSION MANAGEMENT ---
# Dictionary to store history per user: { "session_uuid": deque(...) }
SESSIONS: Dict[str, deque] = {}

def get_history(session_id: str) -> deque:
    """Retrieves or creates the history for a specific session ID."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = deque(maxlen=10)
    return SESSIONS[session_id]

# Helper to format SSE strings
def format_sse(data: str, event: str = "message") -> str:
    msg = f"event: {event}\n"
    for line in data.splitlines():
        msg += f"data: {line}\n"
    msg += "\n"
    return msg

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
        
    # Clear history on page refresh for a clean slate (optional choice)
    if session_id in SESSIONS:
        SESSIONS[session_id].clear()
        
    resp = templates.TemplateResponse("index.html", {"request": request})
    # Set the cookie on the browser so it remembers us
    resp.set_cookie(key="session_id", value=session_id)
    return resp

@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, prompt: str = Form(...)):
    
    stream_id = f"msg-{uuid.uuid4()}"
    
    # Render User Message
    user_message_html = f"""
    <div class="flex justify-end mb-4 animate-fade-in">
        <div class="bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-sm max-w-[80%] shadow-sm">
            <p class="text-sm leading-relaxed">{prompt}</p>
        </div>
    </div>
    """
    
    # Render Bot Placeholder
    # Note: HTMX will automatically send the 'session_id' cookie with this request
    bot_placeholder_html = f"""
    <div id="{stream_id}" 
         hx-ext="sse" 
         sse-connect="/stream?prompt={prompt}&stream_id={stream_id}" 
         sse-swap="message" 
         class="flex justify-start mb-4 animate-fade-in">
        <div class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
            <span id="cursor" class="inline-block w-2 h-5 bg-blue-500 cursor-blink align-middle"></span>
        </div>
    </div>
    """
    
    return user_message_html + bot_placeholder_html

@app.get("/stream")
async def stream_response(prompt: str, stream_id: str, session_id: str = Cookie(None)):
    """
    The 'session_id' is automatically extracted from the browser cookie.
    """
    async def event_generator():
        try:
            # If no session cookie, we can't track history.
            if not session_id:
                yield format_sse("Error: No session found. Refresh page.")
                return

            history = get_history(session_id)

            # 1. Prepare Context
            messages_payload = list(history)
            messages_payload.append({"role": "user", "parts": [prompt]})

            # 2. Call Gemini
            # Instantiate model with the current system instruction
            current_instruction = get_system_instruction()
            # Fallback to a default if empty, or just pass None (Gemini handles None fine usually, but let's be safe)
            if not current_instruction:
                current_instruction = "You are a helpful AI assistant."
                
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=current_instruction
            )
            
            response = model.generate_content(messages_payload, stream=True)
            
            accumulated_text = ""
            
            for chunk in response:
                if chunk.parts:
                    accumulated_text += chunk.text
                    
                    # Live Render: Convert Markdown -> HTML
                    html_content = markdown.markdown(accumulated_text, extensions=['fenced_code'])
                    
                    # Wrap in a div and send
                    # We use the same styling as the final div but without the outer wrapper since we are swapping inner content
                    payload = f"<div>{html_content}</div>"
                    yield format_sse(payload)
                    
                    await asyncio.sleep(0.05)
            
            # 3. Save to Session History
            # (We don't need idempotency check as strictly here if we trust the cookie, 
            # but it doesn't hurt to be safe if you want to keep PROCESSED_IDS logic per session)
            history.append({"role": "user", "parts": [prompt]})
            history.append({"role": "model", "parts": [accumulated_text]})

            # 4. Finalize UI (OOB Swap)
            final_html_content = markdown.markdown(accumulated_text, extensions=['fenced_code'])
            
            final_div = f"""
            <div id="{stream_id}" 
                 hx-swap-oob="outerHTML:#{stream_id}" 
                 class="flex justify-start mb-4">
                <div class="bg-white border border-gray-100 text-gray-800 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm prose prose-sm prose-blue max-w-none">
                    {final_html_content}
                </div>
            </div>
            """
            
            yield format_sse(final_div)
            yield format_sse("", event="close")

        except Exception as e:
            error_div = f"""
            <div id="{stream_id}" 
                 hx-swap-oob="outerHTML:#{stream_id}" 
                 class="flex justify-start mb-4">
                <div class="bg-red-50 border border-red-100 text-red-600 px-5 py-3 rounded-2xl rounded-tl-sm max-w-[90%] shadow-sm text-sm">
                    <strong>Error:</strong> {str(e)}
                </div>
            </div>
            """
            yield format_sse(error_div)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/reset")
async def reset_chat(session_id: str = Cookie(None)):
    if session_id and session_id in SESSIONS:
        SESSIONS[session_id].clear()
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