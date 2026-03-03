import litellm
from litellm import acompletion
from pydantic import BaseModel
import asyncio
import os
import logging
from pathlib import Path
from litellm import Router
import json
import time
import tempfile
litellm.enable_json_schema_validation=True

# Suppress LiteLLM INFO logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
# Also disable LiteLLM's verbose logging
litellm.set_verbose = False

# Create a single shared router instance to avoid callback limit issues
_router = None
_env_loaded = False


def _load_dotenv_once() -> None:
    """Load .env from project root into os.environ (without overrides)."""
    global _env_loaded
    if _env_loaded:
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        _env_loaded = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    _env_loaded = True


def _build_vllm_model_entry():
    """Build a LiteLLM router entry for local OpenAI-compatible vLLM server."""
    _load_dotenv_once()
    api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    vllm_model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    return {
        "model_name": "vllm",
        "litellm_params": {
            "model": f"openai/{vllm_model}",
            "api_base": api_base,
            "api_key": api_key,
        },
    }


def get_litellm_fallback_router():
    """Get or create a shared router instance"""
    global _router
    if _router is None:
        local_entry = _build_vllm_model_entry()
        _router = Router(
            model_list=[
                local_entry,
                #{
                #    "model_name": "claude", 
                #    "litellm_params": {
                #        "model": "anthropic/claude-3-haiku-20240307"}
                #}, 
                {
                    "model_name": "gpt",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini", 
                    }
                },
                {
                    "model_name": "gemini",
                    "litellm_params": {
                        "model": "gemini/gemini-2.0-flash-001", 
                    }
                }
            ],
            fallbacks=[
                {"vllm": ["gpt", "gemini"]},
                {"gpt": ["gemini"]},
                {"gemini": ["gpt"]},
            ],
            num_retries=1,
            max_fallbacks=1, 
        )
    return _router

async def call_llm_with_fallback(str1, 
                                 model_name = "vllm", 
                                 response_format=None
                                 ):
    router = get_litellm_fallback_router()
    kwargs = {"temperature": 0.0}
    result = await call_llm(
        str1, 
        router, 
        model_name, 
        response_format, 
        kwargs
    )
    return result

async def call_llm(str1, 
                   router=None, 
                   model_name="openai/gpt-4o", 
                   response_format=None, 
                   kwargs={}):
    acompletion1 = router.acompletion if router else acompletion
    response = await acompletion1(
        model=model_name,
        messages=[{"role": "user", "content": str1}], 
        response_format=response_format, 
        **kwargs, 
    )

    x = response.choices[0].message.content
    
    # Parse JSON string when using structured output (response_format with json_schema)
    # litellm returns JSON string when using structured output, need to parse it
    if response_format and response_format.get("type") == "json_schema":
        try:
            return json.loads(x)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Failed to parse structured output as JSON: {e}. Returning raw string.")
    
    return x

async def call_llm_stream(str1, 
                         router=None, 
                         model_name="openai/gpt-4o", 
                         response_format=None, 
                         kwargs={}):
    """
    Stream LLM response as async generator yielding chunks.
    
    Yields:
        str: Content chunks from the LLM stream
    """
    acompletion1 = router.acompletion if router else acompletion
    stream = await acompletion1(
        model=model_name,
        messages=[{"role": "user", "content": str1}], 
        response_format=response_format,
        stream=True,
        **kwargs, 
    )
    
    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

async def call_llm_stream_with_fallback(str1, 
                                        model_name="gpt", 
                                        response_format=None):
    """
    Stream LLM response with fallback support.
    
    Yields:
        str: Content chunks from the LLM stream
    """
    router = get_litellm_fallback_router()
    kwargs = {"temperature": 0.0}

    try:
        async for chunk in call_llm_stream(
            str1, 
            router, 
            model_name, 
            response_format, 
            kwargs
        ):
            yield chunk
    except Exception as e:
        # If primary model fails, try fallback
        fallback_model = "gemini" if model_name == "gpt" else "gpt"
        logging.warning(f"Primary model {model_name} failed, trying fallback {fallback_model}: {e}")

        try:
            async for chunk in call_llm_stream(
                str1, 
                router, 
                fallback_model, 
                response_format, 
                kwargs
            ):
                yield chunk
        except Exception as fallback_error:
            logging.error(f"Both models failed: {fallback_error}")
            raise

async def call_llm_with_tools(
    prompt: str,
    tools: list[dict],
    model_name: str = "gpt",
    tool_choice: str = "auto"
) -> dict:
    """
    Call LLM with function calling support via LiteLLM.
    
    Args:
        prompt: User prompt
        tools: List of tool definitions in OpenAI format
        model_name: Model identifier for LiteLLM router
        tool_choice: "auto", "required", or "none"
    
    Returns:
        dict: Contains 'content' and 'tool_calls' (if any)
    """
    router = get_litellm_fallback_router()
    
    messages = [{"role": "user", "content": prompt}]
    start_time = time.time()
    
    # LiteLLM automatically handles tools parameter
    response = await router.acompletion(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.0
    )
    
    latency = time.time() - start_time
    
    message = response.choices[0].message
    
    # Extract function calls if present
    result = {
        "content": message.content or "",
        "tool_calls": []
    }
    
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            result["tool_calls"].append({
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })

    return result