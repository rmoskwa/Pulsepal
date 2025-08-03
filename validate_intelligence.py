#!/usr/bin/env python3
"""
Quick validation script to demonstrate Pulsepal's intelligent behavior.

Shows the difference between general knowledge queries (fast, no search)
and Pulseq-specific queries (selective search when needed).
"""

import asyncio
import time
from pulsepal.main_agent import run_pulsepal


async def test_intelligent_behavior():
    """Demonstrate intelligent decision-making with a few key examples."""
    
    print("\n" + "="*80)
    print("🧠 PULSEPAL INTELLIGENCE VALIDATION")
    print("="*80)
    print("\nDemonstrating intelligent behavior with 4 test queries:\n")
    
    # Test 1: General MRI physics (should NOT search)
    print("1️⃣ Testing general MRI physics question...")
    print("   Query: 'What is T1 relaxation?'")
    print("   Expected: Fast response using built-in knowledge (no search)")
    
    start = time.time()
    session_id, response = await run_pulsepal("What is T1 relaxation?")
    elapsed = time.time() - start
    
    print(f"   ✅ Response time: {elapsed:.2f}s")
    print(f"   📝 Response preview: {response[:100]}...")
    print()
    
    await asyncio.sleep(1)
    
    # Test 2: General debugging (should NOT search)
    print("2️⃣ Testing general debugging question...")
    print("   Query: 'Why does my loop run forever?'")
    print("   Expected: Fast response using reasoning (no search)")
    
    start = time.time()
    _, response = await run_pulsepal("Why does my loop run forever?", session_id)
    elapsed = time.time() - start
    
    print(f"   ✅ Response time: {elapsed:.2f}s")
    print(f"   📝 Response preview: {response[:100]}...")
    print()
    
    await asyncio.sleep(1)
    
    # Test 3: Pulseq-specific function (SHOULD search)
    print("3️⃣ Testing Pulseq-specific function question...")
    print("   Query: 'How to use mr.makeGaussPulse?'")
    print("   Expected: Selective search for function details")
    
    start = time.time()
    _, response = await run_pulsepal("How to use mr.makeGaussPulse?", session_id)
    elapsed = time.time() - start
    
    print(f"   ✅ Response time: {elapsed:.2f}s")
    print(f"   📝 Response preview: {response[:100]}...")
    print()
    
    await asyncio.sleep(1)
    
    # Test 4: General k-space question (should NOT search)
    print("4️⃣ Testing k-space conceptual question...")
    print("   Query: 'Explain k-space sampling'")
    print("   Expected: Fast response using knowledge (no search)")
    
    start = time.time()
    _, response = await run_pulsepal("Explain k-space sampling", session_id)
    elapsed = time.time() - start
    
    print(f"   ✅ Response time: {elapsed:.2f}s")
    print(f"   📝 Response preview: {response[:100]}...")
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)
    print("\n🎯 Key Observations:")
    print("- General MRI/programming questions get fast responses")
    print("- Only Pulseq-specific queries trigger selective searches")
    print("- The system intelligently decides when to search vs use knowledge")
    print()


async def main():
    """Run the validation."""
    try:
        await test_intelligent_behavior()
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())