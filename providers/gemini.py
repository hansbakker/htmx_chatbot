import os
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from typing import List, Dict, Any, AsyncGenerator, Optional
from .base import LLMProvider

import asyncio

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=self.api_key)

    async def upload_file(self, file_path: str, mime_type: str, wait_for_active: bool = True) -> Any:
        """
        Uploads file to Gemini File API.
        """
        print(f"Uploading file to Gemini: {file_path} ({mime_type})")
        uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
        
        if wait_for_active:
            while uploaded_file.state.name == "PROCESSING":
                print("Waiting for file processing...")
                await asyncio.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)
                
            if uploaded_file.state.name == "FAILED":
                raise ValueError("File processing failed.")
                
        return uploaded_file

    async def generate(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> Any:
        
        # 1. Initialize Model
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools
        )
        
        # 2. Generate Content
        # Pass kwargs (like stream=False) to generate_content_async
        return await model.generate_content_async(messages, **kwargs)

    async def stream_chat(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        
        # 1. Initialize Model
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools
        )
        
        # 2. Convert Messages to Gemini Format
        # main.py currently handles history construction manually using genai.protos.Part
        # To make this provider generic, we should accept a generic format and convert it here.
        # BUT, for the first step of refactoring, we can accept the ALREADY FORMATTED messages
        # if we want to minimize change in main.py, OR we move the formatting logic here.
        # Moving formatting logic here is better for abstraction.
        
        # However, main.py's get_history() is very specific to the DB structure.
        # Let's assume 'messages' passed here are already in a format Gemini accepts (list of dicts or Content objects)
        # OR we can standardize on: [{"role": "user", "parts": [...]}, ...]
        
        # For this iteration, let's assume the caller (main.py) passes the list of contents
        # exactly as genai.generate_content expects it.
        # This allows us to reuse the complex history building logic in main.py for now.
        
        # 3. Generate Stream
        try:
            # We use generate_content(stream=True)
            response = model.generate_content(messages, stream=True)
            
            for chunk in response:
                yield chunk
                
        except Exception as e:
            # Re-raise or handle
            raise e
