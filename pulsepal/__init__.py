"""
Pulsepal: Multi-Agent MRI Sequence Programming Assistant

A production-grade AI assistant system for Pulseq MRI sequence programming
with dual specialized agents, RAG integration, and multi-language support.
"""

__version__ = "1.0.0"
__author__ = "Robert Moskwa"

from .main_agent import create_pulsepal_session, pulsepal_agent
from .providers import get_llm_model
from .settings import get_settings, load_settings

__all__ = [
    "create_pulsepal_session",
    "get_llm_model",
    "get_settings",
    "load_settings",
    "pulsepal_agent",
]
