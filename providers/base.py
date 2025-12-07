from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional, Union

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @abstractmethod
    async def stream_chat(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        Streams response from the model.
        
        Args:
            model_name (str): The name of the model to use.
            messages (List[Dict[str, Any]]): The chat history.
            tools (Optional[List[Any]]): List of tool functions.
            system_instruction (Optional[str]): System instruction/persona.
            **kwargs: Additional provider-specific arguments.
            
        Yields:
            Chunks of the response (text or tool calls).
        """
        pass

    @abstractmethod
    async def generate(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Generates a single response (non-streaming).
        
        Args:
            model_name (str): The name of the model to use.
            messages (List[Dict[str, Any]]): The chat history.
            tools (Optional[List[Any]]): List of tool functions.
            system_instruction (Optional[str]): System instruction/persona.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            The response object (provider-specific for now, or standardized).
        """
        pass

    @abstractmethod
    async def upload_file(self, file_path: str, mime_type: str, wait_for_active: bool = True) -> Any:
        """
        Uploads a file to the provider's storage (or prepares it for sending).
        
        Args:
            file_path (str): Local path to the file.
            mime_type (str): MIME type of the file.
            wait_for_active (bool): Whether to wait for the file to be processed.
            
        Returns:
            Any: A reference to the uploaded file (e.g., URI, file ID, or base64 string).
        """
        pass
