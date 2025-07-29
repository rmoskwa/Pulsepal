#!/usr/bin/env python3
"""
Simple CLI interface for Pulsepal multi-agent system.

Usage:
    python run_pulsepal.py "Your question about Pulseq programming"
    python run_pulsepal.py --interactive  # For interactive mode
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add pulsepal to path
sys.path.insert(0, str(Path(__file__).parent))

from pulsepal.main_agent import run_pulsepal, create_pulsepal_session


async def single_query(question: str):
    """Handle a single question."""
    print(f"🔬 Pulsepal: Processing your question...\n")
    
    try:
        session_id, response = await run_pulsepal(question)
        print("🤖 Pulsepal:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        print(f"\n💾 Session ID: {session_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def interactive_mode():
    """Run in interactive mode with session continuity."""
    print("🔬 Pulsepal Interactive Mode")
    print("=" * 40)
    print("Ask me anything about Pulseq MRI sequence programming!")
    print("Type 'quit', 'exit', or Ctrl+C to stop.")
    print("=" * 40)
    
    # Create a persistent session
    session_id, deps = await create_pulsepal_session()
    print(f"💾 Session created: {session_id}\n")
    
    try:
        while True:
            try:
                question = input("\n🙋 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for using Pulsepal!")
                    break
                
                if not question:
                    continue
                
                print("\n🔬 Pulsepal: Thinking...")
                session_id, response = await run_pulsepal(question, session_id)
                
                print("\n🤖 Pulsepal:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using Pulsepal!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")
                
    except Exception as e:
        print(f"❌ Session error: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pulsepal: Multi-Agent MRI Sequence Programming Assistant"
    )
    parser.add_argument(
        "question", 
        nargs="?", 
        help="Your question about Pulseq programming"
    )
    parser.add_argument(
        "-i", "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Pulsepal v1.0.0"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        await interactive_mode()
    elif args.question:
        await single_query(args.question)
    else:
        print("🔬 Pulsepal: Multi-Agent MRI Sequence Programming Assistant")
        print("\nUsage:")
        print('  python run_pulsepal.py "How do I create a spin echo sequence?"')
        print("  python run_pulsepal.py --interactive")
        print("\nFor help: python run_pulsepal.py --help")


if __name__ == "__main__":
    asyncio.run(main())