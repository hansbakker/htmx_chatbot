import os
from google import genai
from google.genai import types
from typing import List, Dict, Any, AsyncGenerator, Optional
from .base import LLMProvider

import asyncio

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})

    async def upload_file(self, file_path: str, mime_type: str, wait_for_active: bool = True) -> Any:
        """
        Uploads file to Gemini File API.
        """
        print(f"Uploading file to Gemini: {file_path} ({mime_type})")
        
        # In google-genai, we use client.files.upload
        uploaded_file = self.client.files.upload(path=file_path, config={'mime_type': mime_type})
        
        if wait_for_active:
            while uploaded_file.state == "PROCESSING":
                print("Waiting for file processing...")
                await asyncio.sleep(1)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
                
            if uploaded_file.state == "FAILED":
                raise ValueError("File processing failed.")
                
        return uploaded_file

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        """
        Converts generic message structure to google-genai types.
        Ensures all parts are wrapped in types.Part.
        """
        converted = []
        for msg in messages:
            role = msg.get('role', 'user')
            parts = msg.get('parts', [])
            
            clean_parts = []
            for part in parts:
                if isinstance(part, str):
                    clean_parts.append(types.Part(text=part))
                else:
                    # Assume it's already a Part-like object or compliant dict
                    clean_parts.append(part)
            
            converted.append(types.Content(role=role, parts=clean_parts))
        return converted

    async def generate(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> Any:
        
        # 1. Convert Messages to GenAI Types
        contents = self._convert_messages(messages)
        
        # 2. Configure Generation
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools
        )
        
        # Merge other kwargs into config if they are valid
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        
        if response.candidates:
            for i, cand in enumerate(response.candidates):
                if cand.content and cand.content.parts:
                    for j, part in enumerate(cand.content.parts):
                        if part.function_call:
                            print(f"DEBUG: [GeminiProvider] Candidate {i} Part {j} - Tool Call: {part.function_call.name}")
                        elif part.text:
                            print(f"DEBUG: [GeminiProvider] Candidate {i} Part {j} - Text: {part.text[:50]}...")
        
        return response

    async def stream_chat(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        
        # 1. Convert Messages to GenAI Types
        contents = self._convert_messages(messages)
        
        # 2. Configure Generation
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools
        )
        
        # Merge other kwargs into config if they are valid
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        try:
            print(f"DEBUG: [GeminiProvider] Starting stream_chat with {len(tools) if tools else 0} tools")
            # We use models.generate_content_stream
            response = self.client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            )
            
            for chunk in response:
                yield chunk
                
        except Exception as e:
            raise e
