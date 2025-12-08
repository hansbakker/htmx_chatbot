import inspect
from typing import Any, Callable, Dict, List, get_type_hints

def function_to_openai_tool(func: Callable) -> Dict[str, Any]:
    """
    Converts a Python function to an OpenAI tool schema.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    hints = get_type_hints(func)
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        # Determine type
        param_type = "string" # Default
        if name in hints:
            hint = hints[name]
            if hint == int:
                param_type = "integer"
            elif hint == float:
                param_type = "number"
            elif hint == bool:
                param_type = "boolean"
            elif hint == list or getattr(hint, "__origin__", None) == list:
                param_type = "array"
            elif hint == dict or getattr(hint, "__origin__", None) == dict:
                param_type = "object"
        
        # Add to properties
        parameters["properties"][name] = {
            "type": param_type,
            "description": f"Parameter {name}" # We could parse docstring for better descriptions
        }
        
        # Handle default values
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": parameters
        }
    }
