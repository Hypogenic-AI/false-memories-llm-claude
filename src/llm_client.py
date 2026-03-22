"""
LLM API client for querying multiple models.
Supports OpenAI and OpenRouter APIs with retry logic and response caching.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from openai import OpenAI

CACHE_DIR = Path("results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(model, messages, temperature):
    """Generate a deterministic cache key for a request."""
    content = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def query_llm(model, messages, temperature=0.0, max_tokens=300, cache=True):
    """
    Query an LLM via OpenAI-compatible API.
    Uses OpenAI API for gpt-* models, OpenRouter for others.
    Returns the response text.
    """
    # Check cache first
    cache_key = get_cache_key(model, messages, temperature)
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["response"]

    # Determine which API to use
    if model.startswith("gpt-") or model.startswith("o"):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        # Use OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(f"No OPENROUTER_API_KEY found for model {model}")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    # Make the API call with retry
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()

            # Cache the response
            if cache:
                with open(cache_path, "w") as f:
                    json.dump({
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "response": text,
                        "timestamp": time.time(),
                    }, f, indent=2)

            return text

        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/5 for {model}: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after 5 attempts for {model}: {e}")
                return f"ERROR: {e}"


def parse_yes_no(response_text):
    """
    Parse a model response to determine if it asserts existence (Yes) or non-existence (No).
    Returns: (bool assertion, float confidence, str reasoning)
    """
    text = response_text.lower().strip()

    # Extract confidence if present (look for number/100 pattern)
    confidence = None
    import re
    conf_match = re.search(r'(?:confidence|certainty)[:\s]*(\d+)', text)
    if conf_match:
        confidence = int(conf_match.group(1)) / 100.0
    else:
        # Look for standalone number near "confidence"
        conf_match = re.search(r'(\d+)\s*/\s*100', text)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100.0

    # Determine yes/no
    # Check first word or first line for the answer
    first_line = text.split('\n')[0].strip()

    # Strong yes indicators
    yes_patterns = [
        text.startswith("yes"),
        first_line.startswith("yes"),
        "yes," in first_line[:20],
        "yes!" in first_line[:20],
        "does exist" in text[:100],
        "it exists" in text[:100],
        "here it is" in text[:50],
        "here's the" in text[:50],
    ]

    # Strong no indicators
    no_patterns = [
        text.startswith("no"),
        first_line.startswith("no"),
        "no," in first_line[:20],
        "no." in first_line[:20],
        "does not exist" in text[:150],
        "doesn't exist" in text[:150],
        "there is no" in text[:100],
        "there isn't" in text[:100],
        "not a real" in text[:100],
        "no such emoji" in text[:100],
    ]

    if any(yes_patterns):
        assertion = True
    elif any(no_patterns):
        assertion = False
    else:
        # Ambiguous — look at overall sentiment
        yes_count = text.count("exist") - text.count("not exist") - text.count("n't exist") - text.count("no ")
        assertion = yes_count > 0

    # Default confidence based on assertion strength
    if confidence is None:
        if "definitely" in text or "certainly" in text or "absolutely" in text:
            confidence = 0.95
        elif "i think" in text or "i believe" in text or "probably" in text:
            confidence = 0.7
        elif "not sure" in text or "uncertain" in text:
            confidence = 0.4
        else:
            confidence = 0.8  # default moderate confidence

    return assertion, confidence, response_text


def check_available_models():
    """Check which API keys are available and return usable models."""
    models = []

    if os.environ.get("OPENAI_API_KEY"):
        models.append(("gpt-4.1", "OpenAI"))
        models.append(("gpt-4.1-mini", "OpenAI"))

    if os.environ.get("OPENROUTER_API_KEY"):
        models.append(("anthropic/claude-sonnet-4", "OpenRouter"))
        models.append(("google/gemini-2.5-flash", "OpenRouter"))

    return models


if __name__ == "__main__":
    available = check_available_models()
    print(f"Available models: {available}")

    # Quick test
    if available:
        model_id, provider = available[0]
        print(f"\nTesting {model_id} via {provider}...")
        resp = query_llm(
            model_id,
            [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": "Does a seahorse emoji exist in Unicode? Answer Yes or No, then briefly explain."},
            ],
            temperature=0.0,
        )
        print(f"Response: {resp}")
        assertion, conf, _ = parse_yes_no(resp)
        print(f"Parsed: assertion={assertion}, confidence={conf}")
