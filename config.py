"""
Shared Azure OpenAI Configuration
This configuration can be imported and used across all projects in the repository.
"""

import os
from pathlib import Path

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://oai-nasco.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
AZURE_KEY = "BVdtDRjC29UejATKda7J4BlembqAJ21L7PzODWc8aNzEsRCgrXRCJQQJ99AKACYeBjFXJ3w3AAABACOGL10p"
API_VERSION = "2025-01-01-preview"

# Parsed endpoint components
AZURE_BASE_URL = "https://oai-nasco.openai.azure.com/openai/deployments/gpt-4o"
AZURE_MODEL = "gpt-4o"
AZURE_RESOURCE_NAME = "oai-nasco"

# Environment variable overrides
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY", AZURE_KEY)
AZURE_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL", AZURE_BASE_URL)
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", API_VERSION)
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL", AZURE_MODEL)


def get_azure_config():
    """
    Returns Azure OpenAI configuration as a dictionary.
    
    Returns:
        dict: Configuration dictionary with base_url, api_key, api_version, and model
    """
    return {
        "base_url": AZURE_BASE_URL,
        "api_key": AZURE_KEY,
        "api_version": API_VERSION,
        "model": AZURE_MODEL,
        "endpoint": AZURE_ENDPOINT
    }


def get_openai_client_config():
    """
    Returns configuration for OpenAI client initialization.
    
    Returns:
        dict: Configuration for OpenAI client with base_url and default_query
    """
    return {
        "base_url": AZURE_BASE_URL,
        "api_key": AZURE_KEY,
        "default_query": {"api-version": API_VERSION}
    }

