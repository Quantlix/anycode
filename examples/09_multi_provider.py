# Demo 09 — Multi-Provider LLM Adapters
# Execute: uv run python examples/09_multi_provider.py
#
# Demonstrates:
#   1. Creating adapters for all 6 supported providers
#   2. Using the provider factory with different configurations
#   3. Sending a simple chat request to each provider
#
# Requires: At least one of the following in .env (or environment):
#   ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
#   AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY,
#   AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT

import asyncio
import os

from dotenv import load_dotenv

from anycode.providers.adapter import create_adapter
from anycode.types import LLMChatOptions, LLMMessage, TextBlock

load_dotenv()


PROVIDERS: list[dict[str, str]] = [
    {"name": "anthropic", "model": "claude-sonnet-4-20250514", "env_key": "ANTHROPIC_API_KEY"},
    {"name": "openai", "model": "gpt-4o-mini", "env_key": "OPENAI_API_KEY"},
    {"name": "google", "model": "gemini-2.5-pro", "env_key": "GOOGLE_API_KEY"},
    {"name": "ollama", "model": "llama3.3:70b", "env_key": ""},
    {"name": "bedrock", "model": "anthropic.claude-3-sonnet-20240229-v1:0", "env_key": "AWS_ACCESS_KEY_ID"},
    {"name": "azure", "model": "gpt-4", "env_key": "AZURE_OPENAI_API_KEY"},
]


async def demo_provider(name: str, model: str, env_key: str) -> None:
    """Try creating an adapter and sending a simple prompt."""
    if env_key and not os.getenv(env_key):
        print(f"  [{name}] SKIPPED — {env_key} not set")
        return

    if name == "ollama":
        print(f"  [{name}] SKIPPED — requires local Ollama server")
        return

    try:
        kwargs: dict[str, str] = {}
        if name == "azure":
            kwargs["endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
        if name == "bedrock":
            kwargs["region"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        adapter = await create_adapter(name, **kwargs)
        messages = [LLMMessage(role="user", content=[TextBlock(text="Say hello in one sentence.")])]
        options = LLMChatOptions(model=model, max_tokens=64)

        result = await adapter.chat(messages, options)
        text = result.content[0].text if result.content else "(empty)"
        print(f"  [{name}] {text[:80]}")
        print(f"           tokens: {result.usage.input_tokens}in / {result.usage.output_tokens}out")
    except Exception as e:
        print(f"  [{name}] ERROR — {e}")


async def main() -> None:
    print("=== Multi-Provider Demo ===\n")
    for provider in PROVIDERS:
        await demo_provider(provider["name"], provider["model"], provider["env_key"])
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
