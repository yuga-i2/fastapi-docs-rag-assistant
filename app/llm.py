"""
LLM provider abstraction.

Supports swapping the generation backend via the LLM_PROVIDER env var,
so the project isn't locked to one vendor (a course requirement/nice-to-have):

    LLM_PROVIDER=openai      (needs OPENAI_API_KEY)      -- default if key present
    LLM_PROVIDER=anthropic   (needs ANTHROPIC_API_KEY)
    LLM_PROVIDER=groq        (needs GROQ_API_KEY)        -- free tier, no card needed
    LLM_PROVIDER=ollama      (needs a local Ollama server, no key)
    LLM_PROVIDER=demo        (no key needed -- extractive fallback so the
                               whole app runs out of the box for grading/demo)

Set LLM_MODEL to override the default model per provider.
"""

from __future__ import annotations

import os


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "groq": "openai/gpt-oss-20b",
    "ollama": "llama3.1",
}


def _detect_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "demo"


def complete(prompt: str, system: str = "", temperature: float = 0.2) -> str:
    """Single entry point used by the rest of the app. Returns plain text."""
    provider = _detect_provider()
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, ""))

    if provider == "openai":
        return _openai_complete(prompt, system, model, temperature)
    if provider == "anthropic":
        return _anthropic_complete(prompt, system, model, temperature)
    if provider == "groq":
        return _groq_complete(prompt, system, model, temperature)
    if provider == "ollama":
        return _ollama_complete(prompt, system, model, temperature)
    return _demo_complete(prompt, system)


def _openai_complete(prompt, system, model, temperature) -> str:
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return resp.choices[0].message.content


def _anthropic_complete(prompt, system, model, temperature) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system or "",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _groq_complete(prompt, system, model, temperature) -> str:
    # Groq's API is OpenAI-compatible, so we reuse the OpenAI SDK
    # pointed at Groq's base URL. Free tier, no card required:
    # https://console.groq.com/keys
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return resp.choices[0].message.content


def _ollama_complete(prompt, system, model, temperature) -> str:
    import requests
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": full_prompt, "stream": False,
              "options": {"temperature": temperature}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _demo_complete(prompt: str, system: str) -> str:
    """
    No API key configured: return a clearly-labeled extractive fallback
    built from the CONTEXT already inserted into the prompt, so the app,
    UI, and monitoring can still be demoed and graded end-to-end without
    any credentials. Swap in a real key to get real generated answers.
    """
    context_marker = "CONTEXT:"
    if context_marker in prompt:
        context = prompt.split(context_marker, 1)[1]
        snippet = context.strip().split("\n\n")[0][:500]
    else:
        snippet = prompt[:500]
    return (
        "[DEMO MODE - no LLM_API_KEY set, showing top retrieved passage "
        "instead of a generated answer]\n\n" + snippet
    )
