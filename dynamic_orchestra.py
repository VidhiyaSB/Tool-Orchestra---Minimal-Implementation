#!/usr/bin/env python3
"""
Dynamic ToolOrchestra - Single Interface for All Usage
"""

import sys
from tool_orchestra import ToolOrchestra

def run_query(query, preference="cost efficient"):
    """Execute single query with streaming output"""
    orchestra = ToolOrchestra()
    
    print(f"Query: {query}")
    print(f"Preference: {preference}")
    print("=" * 50)
    
    # Stream the orchestration process
    trajectory = orchestra.orchestrate_streaming(query, preference)
    
    print("\n" + "=" * 50)
    print(f"RESULT: {len(trajectory.turns)} turns | ${trajectory.total_cost:.2f} | {'SUCCESS' if trajectory.outcome else 'FAILED'}")
    
    if trajectory.outcome:
        final_turn = [turn for turn in trajectory.turns if 'final_answer' in turn]
        if final_turn:
            print(f"ANSWER: {final_turn[0]['final_answer']}")

def interactive_mode():
    """Interactive loop with paper-based preferences"""
    print("ToolOrchestra Dynamic Mode - Paper Implementation")
    print("Type 'quit' to exit")
    
    preferences = {
        "1": "accuracy - prioritize correctness over cost",
        "2": "efficiency - minimize cost and latency", 
        "3": "cost_efficiency - use cheapest appropriate tools",
        "4": "latency_efficiency - minimize response time",
        "5": "specialized_preference - prefer domain-specific models"
    }
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("\nPreference options:")
            for k, v in preferences.items():
                print(f"  {k}. {v}")
            
            pref_choice = input("Select (1-5) or custom: ").strip()
            
            if pref_choice in preferences:
                preference = preferences[pref_choice].split(" - ")[0]
            elif pref_choice:
                preference = pref_choice
            else:
                preference = "efficiency"
            
            run_query(query, preference)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Command line mode
        query = sys.argv[1]
        preference = sys.argv[2] if len(sys.argv) > 2 else "cost efficient"
        run_query(query, preference)
    else:
        # Interactive mode
        interactive_mode()