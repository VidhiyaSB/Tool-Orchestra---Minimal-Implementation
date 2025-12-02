#!/usr/bin/env python3
"""
Generate detailed tool descriptions for NVIDIA Orchestrator model
Based on paper's approach of LLM-generated tool descriptions
"""

def get_tool_descriptions():
    """Return detailed tool descriptions based on capabilities and performance"""
    
    return {
        "web_search": {
            "name": "web_search",
            "cost": 0.01,
            "latency": 2.0,
            "description": "Real-time web search using Tavily API. Excellent for current events, news, stock prices, recent developments. Fast and cheap but limited to publicly available information. Best for: breaking news, current facts, real-time data.",
            "strengths": ["Real-time info", "Current events", "Fast execution", "Very low cost"],
            "weaknesses": ["No reasoning", "Surface-level info", "Depends on web quality"]
        },
        
        "qwen2.5-math-7b": {
            "name": "qwen2.5-math-7b", 
            "cost": 0.10,
            "latency": 3.0,
            "description": "Efficient math specialist model. Handles algebra, calculus, basic proofs, equation solving. Good accuracy for standard math problems. Cost-effective for routine mathematical tasks. Best for: homework-level math, basic calculations, standard formulas.",
            "strengths": ["Fast math solving", "Cost efficient", "Good for standard problems"],
            "weaknesses": ["Limited on complex proofs", "Struggles with novel problems"]
        },
        
        "qwen2.5-math-72b": {
            "name": "qwen2.5-math-72b",
            "cost": 0.50, 
            "latency": 8.0,
            "description": "Advanced math specialist model. Excels at complex proofs, advanced calculus, research-level mathematics. Higher accuracy on difficult problems. More expensive but handles sophisticated mathematical reasoning. Best for: research math, complex proofs, advanced topics.",
            "strengths": ["Complex reasoning", "Research-level math", "High accuracy"],
            "weaknesses": ["More expensive", "Slower execution", "Overkill for simple tasks"]
        },
        
        "qwen2.5-coder-32b": {
            "name": "qwen2.5-coder-32b",
            "cost": 0.30,
            "latency": 5.0, 
            "description": "Coding specialist using Qwen/Qwen2.5-Coder-32B-Instruct. Generates clean, efficient code in Python, JavaScript, Java, C++. Good at algorithms, data structures, debugging. Understands coding best practices. Best for: code generation, debugging, algorithm implementation.",
            "strengths": ["Clean code generation", "Multiple languages", "32B coding specialist", "HF available"],
            "weaknesses": ["Limited domain knowledge", "May need refinement for complex systems"]
        },
        
        "gpt-5-mini": {
            "name": "gpt-5-mini",
            "cost": 0.50,
            "latency": 4.0,
            "description": "Efficient generalist model. Good at creative writing, basic analysis, general questions. Balanced performance for most tasks. Cost-effective alternative to premium models. Best for: simple creative tasks, basic explanations, general queries.",
            "strengths": ["Versatile", "Cost-effective", "Good general knowledge"],
            "weaknesses": ["Less sophisticated than premium models", "May lack depth"]
        },
        
        "claude-opus-4.1": {
            "name": "claude-opus-4.1", 
            "cost": 1.50,
            "latency": 8.0,
            "description": "Creative writing specialist. Excels at poetry, storytelling, creative content, nuanced writing. Strong at understanding context and tone. Premium model for high-quality creative output. Best for: poetry, stories, creative writing, literary analysis.",
            "strengths": ["Excellent creativity", "Nuanced writing", "Strong literary skills"],
            "weaknesses": ["Expensive", "Overkill for simple tasks", "Slower execution"]
        },
        
        "qwen3-32b": {
            "name": "qwen3-32b",
            "cost": 0.40,
            "latency": 6.0,
            "description": "Balanced generalist model. Good reasoning, analysis, and general knowledge. Handles most tasks competently without premium cost. Reliable middle-ground option. Best for: general analysis, explanations, balanced tasks requiring reasoning.",
            "strengths": ["Balanced performance", "Good reasoning", "Moderate cost"],
            "weaknesses": ["Not specialized", "May lack depth in specific domains"]
        },
        
        "qwen3-235b": {
            "name": "qwen3-235b",
            "cost": 1.20,
            "latency": 12.0,
            "description": "Most capable generalist model. Advanced reasoning, complex analysis, sophisticated understanding. Handles difficult tasks requiring deep thinking. Premium performance for challenging queries. Best for: complex analysis, research tasks, sophisticated reasoning.",
            "strengths": ["Advanced reasoning", "Complex analysis", "Sophisticated understanding"],
            "weaknesses": ["Expensive", "Slow execution", "May be overkill"]
        },
        
        "gpt-5": {
            "name": "gpt-5",
            "cost": 2.00,
            "latency": 10.0,
            "description": "Premium reasoning model. Exceptional at complex analysis, research, sophisticated problem-solving. Highest quality output but most expensive. Use for critical tasks requiring best performance. Best for: research analysis, complex reasoning, critical decisions.",
            "strengths": ["Exceptional reasoning", "Highest quality", "Complex problem solving"],
            "weaknesses": ["Most expensive", "Slow", "Overkill for simple tasks"]
        }
    }

def generate_orchestrator_prompt(query, user_preference):
    """Generate enhanced prompt with detailed tool descriptions"""
    
    tools = get_tool_descriptions()
    
    # Create tool list with rich descriptions
    tool_list = []
    for tool_name, info in tools.items():
        tool_list.append(f"""
- {info['name']} (${info['cost']}, ~{info['latency']}s)
  {info['description']}
  Strengths: {', '.join(info['strengths'])}
  Weaknesses: {', '.join(info['weaknesses'])}""")
    
    prompt = f"""You are an AI orchestrator trained to select optimal tools for tasks.

TASK: {query}
USER PREFERENCE: {user_preference}

AVAILABLE TOOLS:
{''.join(tool_list)}

PREFERENCE GUIDELINES:
- "efficiency"/"cost_efficiency": Prioritize lower cost tools when adequate
- "accuracy": Prioritize higher capability tools for best results  
- "latency_efficiency": Prioritize faster tools
- "specialized_preference": Prefer domain-specific tools

Consider:
1. Task complexity and requirements
2. Tool specialization vs generalist capability
3. Cost-benefit analysis for this specific task
4. User's stated preference

Make an intelligent decision balancing task needs with user preference.

Respond: {{"tool": "exact_name", "reasoning": "detailed_explanation"}}"""
    
    return prompt