#!/usr/bin/env python3
"""
Test script for RAG integration in Pulsepal.

This script tests the basic functionality of the RAG services
including embeddings, Supabase connection, and search capabilities.
"""

import asyncio
import logging
from pulsepal.embeddings import create_embedding, get_embedding_service
from pulsepal.supabase_client import get_supabase_client
from pulsepal.rag_service import get_rag_service
from pulsepal.web_search import get_web_search_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_embeddings():
    """Test embedding service functionality."""
    print("\n=== Testing Embeddings Service ===")
    try:
        # Test single embedding
        test_text = "How do I create a spin echo sequence in Pulseq?"
        embedding = create_embedding(test_text)
        print(f"✓ Created embedding for text: '{test_text[:50]}...'")
        print(f"  Embedding dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        
        # Test that it's not all zeros
        if all(v == 0.0 for v in embedding):
            print("✗ Warning: Embedding is all zeros!")
        else:
            print("✓ Embedding contains non-zero values")
            
        return True
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")
        return False


async def test_supabase_connection():
    """Test Supabase client connection."""
    print("\n=== Testing Supabase Connection ===")
    try:
        client = get_supabase_client()
        print("✓ Supabase client initialized")
        
        # Test getting sources
        sources = client.get_available_sources()
        print(f"✓ Retrieved {len(sources)} sources from database")
        
        if sources:
            print(f"  Example source: {sources[0].get('source_id', 'Unknown')}")
            
        return True
    except Exception as e:
        print(f"✗ Supabase test failed: {e}")
        return False


async def test_rag_search():
    """Test RAG search functionality."""
    print("\n=== Testing RAG Search ===")
    try:
        rag_service = get_rag_service()
        
        # Test documentation search
        query = "gradient echo sequence"
        print(f"\nSearching for: '{query}'")
        results = rag_service.perform_rag_query(query, match_count=3)
        print("✓ Documentation search completed")
        print(f"Results preview:\n{results[:500]}...")
        
        # Test code search
        code_results = rag_service.search_code_examples(query, match_count=2)
        print("\n✓ Code search completed")
        print(f"Code results preview:\n{code_results[:500]}...")
        
        return True
    except Exception as e:
        print(f"✗ RAG search test failed: {e}")
        return False


async def test_web_search():
    """Test web search functionality."""
    print("\n=== Testing Web Search ===")
    try:
        web_service = get_web_search_service()
        
        # Test MRI information search
        query = "MRI pulse sequence timing"
        results = web_service.search_mri_information(query, max_results=3)
        print("✓ Web search completed (mock results)")
        print(f"Results preview:\n{results[:500]}...")
        
        return True
    except Exception as e:
        print(f"✗ Web search test failed: {e}")
        return False


async def test_pulsepal_tools():
    """Test Pulsepal tools integration."""
    print("\n=== Testing Pulsepal Tools ===")
    try:
        # Create a minimal test session
        from pulsepal.main_agent import create_pulsepal_session
        
        session_id, deps = await create_pulsepal_session()
        print(f"✓ Created Pulsepal session: {session_id}")
        print(f"✓ RAG initialized: {deps.rag_initialized}")
        
        return True
    except Exception as e:
        print(f"✗ Pulsepal tools test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Starting RAG Integration Tests for Pulsepal")
    print("=" * 50)
    
    # Check environment variables
    import os
    print("\nChecking environment variables:")
    env_vars = {
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "SUPABASE_URL": bool(os.getenv("SUPABASE_URL")),
        "SUPABASE_KEY": bool(os.getenv("SUPABASE_KEY")),
        "BGE_MODEL_PATH": os.getenv("BGE_MODEL_PATH", "Using default")
    }
    
    for var, status in env_vars.items():
        if var == "BGE_MODEL_PATH":
            print(f"  {var}: {status}")
        else:
            print(f"  {var}: {'✓ Set' if status else '✗ Not set'}")
    
    # Run tests
    tests = [
        ("Embeddings", test_embeddings),
        ("Supabase", test_supabase_connection),
        ("RAG Search", test_rag_search),
        ("Web Search", test_web_search),
        ("Pulsepal Tools", test_pulsepal_tools)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! RAG integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())