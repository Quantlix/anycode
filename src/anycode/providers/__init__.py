from anycode.providers.adapter import create_adapter

__all__ = ["create_adapter"]

# Adapter classes are lazy-imported via create_adapter() to avoid
# requiring optional SDKs at import time.  They can also be imported
# directly when the corresponding extras group is installed:
#
#   from anycode.providers.google import GeminiAdapter
#   from anycode.providers.ollama import OllamaAdapter
#   from anycode.providers.bedrock import BedrockAdapter
#   from anycode.providers.azure import AzureOpenAIAdapter
