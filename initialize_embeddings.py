#!/usr/bin/env python3
"""
Initialize the embedding model to ensure it's ready for use.

Run this script once after installation to pre-load the BGE model
and verify the embedding service is working correctly.
"""

import logging
import time
from pulsepal.embeddings import get_embedding_service, create_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize and test the embedding service."""
    print("Initializing Pulsepal Embedding Service")
    print("=" * 50)
    
    try:
        # Time the initialization
        start_time = time.time()
        
        print("\n1. Loading embedding service...")
        service = get_embedding_service()
        
        init_time = time.time() - start_time
        print(f"✓ Embedding service loaded in {init_time:.2f} seconds")
        
        # Test the service
        print("\n2. Testing embedding creation...")
        test_text = "How do I create a spin echo sequence in Pulseq?"
        
        start_time = time.time()
        embedding = create_embedding(test_text)
        embed_time = time.time() - start_time
        
        print(f"✓ Created test embedding in {embed_time:.2f} seconds")
        print(f"  Embedding dimensions: {len(embedding)}")
        
        # Verify it's not all zeros
        if all(v == 0.0 for v in embedding):
            print("⚠️  Warning: Embedding is all zeros!")
        else:
            non_zero_count = sum(1 for v in embedding if v != 0.0)
            print(f"✓ Embedding contains {non_zero_count} non-zero values")
        
        print("\n✅ Embedding service is ready for use!")
        print("\nYou can now run Pulsepal without initialization delays.")
        
    except Exception as e:
        print(f"\n❌ Error initializing embedding service: {e}")
        print("\nPlease check:")
        print("1. BGE_MODEL_PATH is set correctly in your .env file")
        print("2. The model files exist at the specified path")
        print("3. You have installed all required dependencies (torch, transformers)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())