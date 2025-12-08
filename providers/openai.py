import os
import json
import base64
from typing import List, Dict, Any, AsyncGenerator, Optional
from .base import LLMProvider
from .utils import function_to_openai_tool
import openai
from google.generativeai import protos # Needed to parse history

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.default_model = model_name

    async def upload_file(self, file_path: str, mime_type: str, wait_for_active: bool = True) -> Any:
        """
        OpenAI handles images via base64 encoded strings in the message content (Vision).
        For other files (PDF, CSV), we'll currently return a placeholder or raw text if possible,
        but for now we focus on images.
        """
        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return {"mime_type": mime_type, "data": encoded_string}
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return {"error": str(e)}

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        openai_messages = []
        # Queue of (id, name) for pending tool calls that need responses
        pending_tool_calls = [] 
        
        # Debug logging
        # print(f"DEBUG: Converting {len(messages)} messages...")
        
        for msg in messages:
            role = msg["role"]
            parts = msg["parts"]
            
            # Map 'model' to 'assistant'
            if role == "model":
                role = "assistant"
            
            # If we encounter a non-tool message (User or Assistant) and we have pending tool calls,
            # we must resolve them to avoid "dangling tool call" errors.
            if role in ["user", "assistant"] and pending_tool_calls:
                # print(f"DEBUG: Resolving dangling calls before new {role} message: {pending_tool_calls}")
                # We have pending calls but we're starting a new turn.
                # This means the previous tool calls were interrupted or failed.
                # We must inject dummy tool responses or strip the tool calls.
                # Injecting dummy responses is safer for preserving the flow.
                for call_id, call_name in pending_tool_calls:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Tool {call_name} execution interrupted or result missing."
                    })
                pending_tool_calls = []

            # OpenAI 'content' can be a string or a list of parts (for multimodal)
            message_content_parts = []
            
            tool_calls = []
            
            for part in parts:
                if isinstance(part, str):
                    if part.strip():
                        message_content_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict) and "data" in part and "mime_type" in part:
                    # Specific handling for our upload_file return dictionary
                    mime = part["mime_type"]
                    b64_data = part["data"]
                    
                    if mime.startswith("image/"):
                         message_content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_data}"
                            }
                        })
                    elif mime.startswith("text/") or mime in ["application/json", "application/xml", "application/javascript", "application/x-python-code"]:
                        try:
                            # Decode text content and inject
                            text_content = base64.b64decode(b64_data).decode('utf-8')
                            # Truncate if too huge? Let's rely on model limits for now or trunc at reasonable limit (e.g. 100k chars)
                            if len(text_content) > 100000:
                                text_content = text_content[:100000] + "\n... (truncated)"
                                
                            message_content_parts.append({
                                "type": "text", 
                                "text": f"\n[Uploaded File Content ({mime})]:\n```\n{text_content}\n```\n"
                            })
                        except Exception as e:
                            print(f"Error decoding text file for OpenAI: {e}")
                            message_content_parts.append({"type": "text", "text": f"[Error decoding text file ({mime})]"})
                    else:
                        message_content_parts.append({"type": "text", "text": f"[File ({mime}) - Not supported in OpenAI Chat]"})

                else:
                    # Support both protos.Part and OpenAIPartAdapter (duck typing)
                    fc = getattr(part, "function_call", None)
                    fr = getattr(part, "function_response", None)
                    fd = getattr(part, "file_data", None)
                    
                    if fc:
                        # Generate a synthetic ID
                        # Use a stable index based on total messages + current batch
                        call_id = f"call_{fc.name}_{len(openai_messages)}_{len(tool_calls)}"
                        
                        # print(f"DEBUG: Generated tool call ID: {call_id} for {fc.name}")
                        
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(dict(fc.args)) if hasattr(fc, "args") else "{}"
                            }
                        })
                        pending_tool_calls.append((call_id, fc.name))
                        
                    elif fr:
                        pass # Handled below
                    elif fd:
                        # File data - Handle OpenAI Vision
                        uri = getattr(fd, "file_uri", "unknown")
                        mime = getattr(fd, "mime_type", "application/octet-stream")
                        
                        if mime.startswith("image/"):
                            try:
                                # If uri is a local path (starts with /), read and encode
                                # If it was a web URL, we could pass it directly, but main.py gives local paths
                                if os.path.exists(uri):
                                    with open(uri, "rb") as img_f:
                                        b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                                    
                                    message_content_parts.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64_data}"
                                        }
                                    })
                                else:
                                    message_content_parts.append({"type": "text", "text": f"[Image file missing: {uri}]"})
                            except Exception as e:
                                print(f"Error processing image for OpenAI: {e}")
                                message_content_parts.append({"type": "text", "text": f"[Error reading image: {uri}]"})
                        else:
                            message_content_parts.append({"type": "text", "text": f"[File: {uri} ({mime}) - Not supported in OpenAI Vision]"})
            
            if role == "function":
                # OpenAI uses role='tool'
                role = "tool"
                
                # We need to match this response to a pending tool call
                if pending_tool_calls:
                    # Match FIFO
                    tool_call_id, tool_name = pending_tool_calls.pop(0)
                    # print(f"DEBUG: Matched response to call ID: {tool_call_id}")
                else:
                    # Orphaned response?
                    tool_call_id = "call_unknown_orphan"
                    # print(f"DEBUG: Orphaned tool response found! ID: {tool_call_id}")
                
                # Extract content
                tool_content = ""
                for part in parts:
                    if isinstance(part, str):
                        tool_content += part
                    else:
                        fr = getattr(part, "function_response", None)
                        if fr:
                            try:
                                # Helper for recursive dict conversion of MapComposite/RepeatedComposite
                                def _to_serializable(obj):
                                    if hasattr(obj, "items"):
                                        return {k: _to_serializable(v) for k, v in obj.items()}
                                    elif isinstance(obj, (list, tuple)):
                                        return [_to_serializable(v) for v in obj]
                                    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                                        return [_to_serializable(v) for v in obj]
                                    return obj

                                # Handle both protos and adapters
                                resp = getattr(fr, "response", None)
                                if resp:
                                    resp_dict = _to_serializable(resp)
                                    tool_content += json.dumps(resp_dict)
                                else:
                                    tool_content += "Tool executed."
                            except Exception as e:
                                print(f"Error serializing tool response: {e}")
                                tool_content += "Tool executed successfully"
                
                if not tool_content:
                    tool_content = "Tool executed."
                
                # print(f"DEBUG: Tool response content: {tool_content[:50]}...")

                openai_messages.append({
                    "role": role,
                    "tool_call_id": tool_call_id,
                    "content": tool_content
                })
                
            else:
                # User or Assistant
                msg_obj = {"role": role}
                
                # Ensure content is present and is a string
                # OpenAI requires content to be a string (can be empty) for User messages.
                # For Assistant messages, it can be null ONLY if tool_calls are present.
                
                if role == "user":
                    # User message MUST have content (string or list of parts)
                    msg_obj["content"] = message_content_parts if message_content_parts else " " 
                elif role == "assistant":
                    if tool_calls:
                        msg_obj["tool_calls"] = tool_calls
                        # Content can be omitted or null if tool_calls exist
                        if message_content_parts:
                            msg_obj["content"] = message_content_parts
                    else:
                        # Assistant message without tool calls MUST have content
                        msg_obj["content"] = message_content_parts if message_content_parts else " "
                
                openai_messages.append(msg_obj)
        
        # Final cleanup: If we have pending calls at the VERY END, we must also resolve them
        # because the model cannot reply if there are outstanding tool calls.
        if pending_tool_calls:
             for call_id, call_name in pending_tool_calls:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"Tool {call_name} execution interrupted or result missing."
                })
        
        # Debug logging
        # print(f"DEBUG: OpenAI Messages Payload: {json.dumps(openai_messages, indent=2)}")
        return openai_messages

    async def generate(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> Any:
        
        # Convert messages
        openai_messages = self._convert_messages(messages)
        
        # Add system instruction
        if system_instruction:
            openai_messages.insert(0, {"role": "system", "content": system_instruction})
            
        # Convert tools
        openai_tools = None
        if tools:
            openai_tools = [function_to_openai_tool(t) for t in tools]
            
        # Call OpenAI
        # Map model names if necessary
        if "gemini" in model_name:
            model_name = self.default_model # Fallback to default if gemini model passed
            
        # Remove stream from kwargs if present to avoid duplication
        kwargs.pop("stream", None)
        
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            tools=openai_tools,
            stream=False,
            **kwargs
        )
        
        return OpenAIResponseAdapter(response)
        
    async def stream_chat(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        
        # Convert messages
        openai_messages = self._convert_messages(messages)
        
        # Add system instruction
        if system_instruction:
            openai_messages.insert(0, {"role": "system", "content": system_instruction})
            
        # Convert tools
        openai_tools = None
        if tools:
            openai_tools = [function_to_openai_tool(t) for t in tools]
            
        # Call OpenAI
        if "gemini" in model_name:
            model_name = self.default_model
            
        # Remove stream from kwargs if present to avoid duplication
        kwargs.pop("stream", None)
        
        stream = await self.client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            tools=openai_tools,
            stream=True,
            **kwargs
        )
        
        accumulated_tool_calls = {} # index -> {id, name, args}
        
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
                
            # Handle Text
            if delta.content:
                yield OpenAIChunkAdapter(chunk)
                
            # Handle Tool Calls (Accumulate)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {"id": "", "name": "", "args": ""}
                    
                    if tc.id:
                        accumulated_tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated_tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            accumulated_tool_calls[idx]["args"] += tc.function.arguments
                            
        # Yield accumulated tool calls as complete objects
        for idx, tc_data in accumulated_tool_calls.items():
            yield OpenAIFinalToolCallAdapter(tc_data)

class OpenAIResponseAdapter:
    def __init__(self, response):
        self.response = response
        self.candidates = [OpenAICandidateAdapter(response.choices[0])]
        # Helper text property for non-streaming full response
        self.text = self.candidates[0].content.parts[0].text if self.candidates[0].content.parts else ""

class OpenAICandidateAdapter:
    def __init__(self, choice):
        self.content = OpenAIContentAdapter(choice.message)

class OpenAIContentAdapter:
    def __init__(self, message):
        self.parts = []
        if message.content:
            self.parts.append(OpenAIPartAdapter(text=message.content))
        if message.tool_calls:
            for tc in message.tool_calls:
                self.parts.append(OpenAIPartAdapter(function_call=tc))

class OpenAIPartAdapter:
    def __init__(self, text=None, function_call=None):
        self.text = text
        if function_call:
            self.function_call = OpenAIFunctionCallAdapter(function_call)
        else:
            self.function_call = None

class OpenAIFunctionCallAdapter:
    def __init__(self, tool_call):
        # Handle both OpenAI object and dict (from accumulator)
        if isinstance(tool_call, dict):
            self.name = tool_call["name"]
            try:
                self.args = json.loads(tool_call["args"])
            except json.JSONDecodeError:
                print(f"Error decoding JSON args: {tool_call['args']}")
                self.args = {}
        else:
            self.name = tool_call.function.name
            self.args = json.loads(tool_call.function.arguments)

class OpenAIChunkAdapter:
    def __init__(self, chunk):
        self.chunk = chunk
        self.candidates = [OpenAIChunkCandidateAdapter(chunk.choices[0])] if chunk.choices else []
        
    @property
    def text(self):
        if self.candidates and self.candidates[0].content.parts:
            return self.candidates[0].content.parts[0].text
        return ""
        
    @property
    def function_call(self):
        # Chunks don't yield function calls directly anymore, 
        # we yield OpenAIFinalToolCallAdapter instead.
        return None

class OpenAIChunkCandidateAdapter:
    def __init__(self, choice):
        self.content = OpenAIChunkContentAdapter(choice.delta)

class OpenAIChunkContentAdapter:
    def __init__(self, delta):
        self.parts = []
        if delta.content:
            self.parts.append(OpenAIPartAdapter(text=delta.content))
        # We ignore tool_calls here because we handle them in the stream loop

class OpenAIFinalToolCallAdapter:
    """Adapts a fully accumulated tool call to look like a response part."""
    def __init__(self, tc_data):
        self.tc_data = tc_data
        
    @property
    def text(self):
        return None
        
    @property
    def function_call(self):
        return OpenAIFunctionCallAdapter(self.tc_data)
