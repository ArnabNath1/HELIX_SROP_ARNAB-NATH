import asyncio
import json
import time
import httpx
import sys
from typing import List, Dict

BASE_URL = "http://127.0.0.1:8000"

EVAL_DATA = [
    {"query": "How do I set up a runner?", "expected": "knowledge"},
    {"query": "What is the rotation policy for deploy keys?", "expected": "knowledge"},
    {"query": "Show my recent builds", "expected": "account"},
    {"query": "What is my current plan tier?", "expected": "account"},
    {"query": "I need to talk to a human", "expected": "support"},
    {"query": "This is too complex, open a ticket", "expected": "support"},
    {"query": "Hi there!", "expected": "smalltalk"},
    {"query": "What's the weather like?", "expected": "smalltalk"},
]

async def run_eval():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Create a session for eval
        try:
            resp = await client.post(f"{BASE_URL}/v1/sessions", json={"user_id": "eval_user", "plan_tier": "pro"})
            resp.raise_for_status()
            session_id = resp.json()["session_id"]
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            sys.exit(1)

        print(f"Starting Evaluation on {len(EVAL_DATA)} queries...")
        print("-" * 50)
        
        correct = 0
        results = []

        for item in EVAL_DATA:
            query = item["query"]
            expected = item["expected"]
            
            # Sleep to avoid rate limits
            await asyncio.sleep(2)
            
            start = time.monotonic()
            try:
                resp = await client.post(f"{BASE_URL}/v1/chat/{session_id}", json={"content": query})
                resp.raise_for_status()
                data = resp.json()
                actual = data["routed_to"]
                latency = int((time.monotonic() - start) * 1000)
                
                is_correct = (actual == expected)
                if is_correct:
                    correct += 1
                
                status = "[PASS]" if is_correct else "[FAIL]"
                print(f"{status} Query: {query[:40]:<40} | Expected: {expected:<10} | Actual: {actual:<10} | {latency}ms")
                
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": actual,
                    "correct": is_correct,
                    "latency": latency
                })
            except Exception as e:
                print(f"[ERROR] on query '{query}': {e}")

        accuracy = (correct / len(EVAL_DATA)) * 100 if EVAL_DATA else 0
        print("-" * 50)
        print(f"Final Accuracy: {accuracy:.1f}% ({correct}/{len(EVAL_DATA)})")
        
        # Save results to a file
        with open("eval_results.json", "w") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_eval())
