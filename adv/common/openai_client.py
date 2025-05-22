import json
import asyncio
import hashlib
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from diskcache import Cache
from openai import AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param

load_dotenv(find_dotenv())

class AsyncCache:
    def __init__(self, cache_path):
        self._cache = Cache(cache_path)
    
    async def get(self, key):
        return await asyncio.to_thread(self._cache.get, key)
    
    async def set(self, key, value, expire=None):
        return await asyncio.to_thread(self._cache.set, key, value, expire=expire)

async_cache = AsyncCache("./async_cache")

async def async_call_openai(client, messages, model, system_prompt, response_format, max_tokens, temperature):
    """
    Makes an asynchronous OpenAI API call with caching.

    Args:
        client (AsyncOpenAI): OpenAI client.
        messages (list): List of messages for the chat completion.
        model (str): Model to use.
        system_prompt (str): System prompt for the model.
        response_format (Union[BaseModel, str]): Format of the response.
        max_tokens (int): Maximum number of tokens in the response.
        temperature (float): Sampling temperature.

    Returns:
        str: The content of the response message.
    """
    
    if not isinstance(response_format, str):
        response_format = type_to_response_format_param(response_format)
    
    # Create a unique cache key using a hash of the inputs
    cache_key = hashlib.sha256(
        json.dumps({
            "messages": messages,
            "model": model,
            "system_prompt": system_prompt,
            "response_format": response_format,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, sort_keys=True).encode()
    ).hexdigest()

    # Check if the result is already in the cache
    cached_response = await async_cache.get(cache_key)
    if cached_response is not None:
        return cached_response

    # Prepare the API call
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages if system_prompt is not None else messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )
    out = response.choices[0].message.content

    # Save the result to the cache -- no expiration so it always stays in the cache
    await async_cache.set(cache_key, out, expire=None)
    return out
