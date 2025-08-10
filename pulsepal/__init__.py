"""
Pulsepal: Multi-Agent MRI Sequence Programming Assistant

A production-grade AI assistant system for Pulseq MRI sequence programming
with dual specialized agents, RAG integration, and multi-language support.
"""

__version__ = "1.0.0"
__author__ = "Robert Moskwa"

from .main_agent_v2 import pulsepal_agent, create_pulsepal_session
from .settings import load_settings, get_settings
from .providers import get_llm_model

__all__ = [
    "pulsepal_agent",
    "create_pulsepal_session",
    "load_settings",
    "get_settings",
    "get_llm_model",
]
