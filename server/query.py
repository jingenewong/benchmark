import os
import json
import re
import time
import asyncio
import httpx
from typing import Any, Dict
import datetime
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

TIMEOUT = 7200
NUM_RETRIES = 10
_REGISTRY: Dict[str, Any] = {}

def register_model(model: str):
    def decorator(fn):
        _REGISTRY[model] = fn
        return fn
    return decorator

# ------------------------ Extract value ------------------------

def extract_value(text: str, length: int):
    if length > 0:
        match = re.search(r'\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*\]', text)
        return [float(x.strip()) for x in match.group(0)[1:-1].split(',')][:length]
    else:
        match = re.search(r'-?\d+(?:\.\d+)?', text)
        return float(match.group()) 

def check_type(output, length):
    if length == 0:
        return isinstance(output, (float, int))
    return isinstance(output, list) and all(isinstance(x, (float, int)) for x in output)

# ------------------------ OpenAI (Tool-calling) ------------------------

@register_model("o1-low")
@register_model("o1-medium")
@register_model("o1-high")
@register_model("o3-low")
@register_model("o3-medium")
@register_model("o3-high")
@register_model("o3-mini-low")
@register_model("o3-mini-medium")
@register_model("o3-mini-high")
@register_model("o4-mini-low")
@register_model("o4-mini-medium")
@register_model("o4-mini-high")
async def _openai_with_tool(model: str, prompt: str, length=0, **kwargs) -> Any:
    tools = [{
        "type": "function",
        "function": {
            "name": "return_array",
            "description": f"Return an array of exactly {max(length, 1)} decimal numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": max(length, 1),
                        "maxItems": max(length, 1)
                    }
                },
                "required": ["numbers"]
            }
        }
    }]
    reasoning = model.split("-")[-1]
    model_base = model[: -len(reasoning) - 1] if reasoning in ["low", "medium", "high"] else model
    payload = {
        "model": model_base,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": "auto"
    }
    if reasoning in ["low", "medium", "high"]:
        payload["reasoning_effort"] = reasoning

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        s = time.time()
        
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        args_str = data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        res = json.loads(args_str)['numbers']
        
        usage = data.get("usage", {})
        
        return res if length != 0 else res[0]

# ------------------------ OpenAI basic + DeepSeek ------------------------

@register_model("o1-mini")
async def _openai_basic(model: str, prompt: str, length=0, **kwargs) -> Any:
    reasoning = model.split("-")[-1]
    model_base = model[: -len(reasoning) - 1] if reasoning in ["low", "medium", "high"] else model
    payload = {
        "model": model_base,
        "messages": [{"role": "user", "content": prompt}]
    }
    if reasoning in ["low", "medium", "high"]:
        payload["reasoning_effort"] = reasoning

    key = os.getenv("DEEPSEEK_API_KEY") if model.startswith("deepseek") else os.getenv("OPENAI_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions" if model.startswith("deepseek") else "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return extract_value(text, length)

@register_model("deepseek-reasoner")
async def call_deepseek(model: str, prompt: str, length=0, **kwargs) -> Any:
    model = "deepseek/deepseek-r1"
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_value(response.choices[0].message.content, length)

# ------------------------ Gemini ------------------------
_cache_registry = {}
_registry_lock = asyncio.Lock()
CACHE_TTL_SECONDS = 3600 
MIN_REFRESH_WINDOW = 600


_cache_registry: Dict[str, datetime.datetime] = {}
_registry_lock = asyncio.Lock()

async def ensure_cached_prompt(prompt: str, model: str) -> str:
    cache_id = str(abs(hash(prompt)))
    api_key = os.getenv("GEMINI_API_KEY")
    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = now + datetime.timedelta(seconds=CACHE_TTL_SECONDS)

    # fast path: already fresh
    async with _registry_lock:
        if cache_id in _cache_registry and (_cache_registry[cache_id] - now).total_seconds() > MIN_REFRESH_WINDOW:
            return cache_id

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    create_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:cachedContents.create"
    create_body = {
        "id": cache_id,
        "ttl": f"{CACHE_TTL_SECONDS}s",
        "contents": [{"parts": [{"text": prompt}]}]
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            # try to create
            resp = await client.post(create_url, headers=headers, params=params, json=create_body)
            resp.raise_for_status()
            print(f"✅ Created cache {cache_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                # already exists → update TTL
                update_url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents/{cache_id}:update"
                update_body = {"config": {"expireTime": expires_at.isoformat()}}
                upd = await client.post(update_url, headers=headers, params=params, json=update_body)
                upd.raise_for_status()
                print(f"🔁 Refreshed TTL for {cache_id}")
            else:
                # some other error
                print(f"❌ Cache create error: {e.response.text}")
                raise

    # record the new expiry
    async with _registry_lock:
        _cache_registry[cache_id] = expires_at

    return cache_id


@register_model("gemini-1.5-flash")
@register_model("gemini-2.0-flash")
@register_model("gemini-2.5-pro-exp-03-25")
@register_model("gemini-2.0-flash-lite-preview-02-05")
async def _gemini_basic(model: str, prompt: str, length=0, **kwargs) -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing from environment")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}        
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    
    #TODO: get cache working for gemini
    # cache_id = await ensure_cached_prompt(prompt, model)
    # body["context"] = { "cached": { "id": cache_id} }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, headers=headers, params=params, json=body)
        resp.raise_for_status()        
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return extract_value(text, length)

# ------------------------ Anthropic ------------------------

@register_model("claude-3-7-sonnet-20250219")
async def _claude_basic(model: str, prompt: str, length=0, **kwargs) -> Any:
    headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    body = {
        "model": model,
        "max_tokens": 64000,
        "system": [
            {
                "type": "text",
                "text": prompt,
                "cache_control": { "type": "ephemeral" }
            }
        ],
        "messages": [
            { "role": "user", "content": "Give answer only" }
        ]
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        return extract_value(text, length)


# ------------------------ Ask wrapper ------------------------

async def ask_model(model: str, prompt: str, cache, return_length=0) -> Any:
    handler = _REGISTRY.get(model)
    if not handler:
        raise ValueError(f"No handler registered for model {model}")

    result = None
    for attempt in range(NUM_RETRIES):
        await asyncio.sleep(attempt * 60)
        try:
            result = await handler(model, prompt, return_length)
            if check_type(result, return_length):
                break
            raise TypeError(f"Check Type failed")
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                await asyncio.sleep(attempt)
            print(f"Error calling model {model} on attempt {attempt+1}: {e}")

    return result
