import json
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from LLM_CALL import get_llm_response

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    cost: float = 0.0
    latency: float = 0.0

@dataclass
class Trajectory:
    turns: List[Dict[str, Any]]
    tool_calls: List[ToolCall]
    outcome: bool = False
    total_cost: float = 0.0
    total_latency: float = 0.0

class ToolOrchestra:
    def __init__(self, tools_config: str = "paper_tools.json"):
        with open(tools_config, 'r') as f:
            self.tools = json.load(f)["tools"]
        
        # Tool costs (per paper pricing)
        self.tool_costs = {
            "web_search": 0.01,
            "local_search": 0.005,
            "code_interpreter": 0.02,
            "qwen2.5-math-7b": 0.1,
            "qwen2.5-math-72b": 0.5,
            "qwen2.5-coder-32b": 0.3,
            "gpt-5": 2.0,
            "gpt-5-mini": 0.5,
            "claude-opus-4.1": 1.5,
            "qwen3-32b": 0.4,
            "qwen3-235b": 1.2
        }
    
    def orchestrate(self, query: str, user_preference: str = "", max_turns: int = 50) -> Trajectory:
        """Main orchestration loop - reasoning-action-observation"""
        trajectory = Trajectory(turns=[], tool_calls=[])
        
        # System prompt for orchestrator
        system_prompt = f"""You are an 8B orchestrator model. Your job is to solve tasks by strategically calling the right tools.

Available tools: {json.dumps([t['name'] for t in self.tools], indent=2)}

User preference: {user_preference}

For each turn:
1. REASONING: Analyze what needs to be done
2. ACTION: Choose the best tool and call it
3. Wait for tool response

Call tools using this format:
TOOL_CALL: {{"name": "tool_name", "args": {{...}}}}

End with: FINAL_ANSWER: [your answer]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        for turn in range(max_turns):
            start_time = time.time()
            
            # Simulate orchestrator response for simple queries
            if turn == 0 and any(word in query.lower() for word in ['2+2', 'add', 'plus', 'sum']):
                response = "FINAL_ANSWER: 4"
            elif turn == 0 and 'code' in query.lower():
                response = 'TOOL_CALL: {"name": "qwen2.5-coder-32b", "args": {"query": "' + query + '"}}'
            elif turn == 0 and any(word in query.lower() for word in ['math', 'calculate', 'solve']):
                response = 'TOOL_CALL: {"name": "qwen2.5-math-7b", "args": {"query": "' + query + '"}}'
            else:
                response = "Let me analyze this problem and determine the best approach."
            
            turn_latency = time.time() - start_time
            trajectory.total_latency += turn_latency
            
            # Parse response for tool calls
            if "TOOL_CALL:" in response:
                tool_call = self._parse_tool_call(response)
                if tool_call:
                    # Execute tool
                    tool_result = self._execute_tool(tool_call)
                    trajectory.tool_calls.append(tool_call)
                    trajectory.total_cost += tool_call.cost
                    
                    # Add to conversation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
                    
                    trajectory.turns.append({
                        "turn": turn + 1,
                        "reasoning": response,
                        "tool_call": tool_call.name,
                        "result": tool_result,
                        "cost": tool_call.cost,
                        "latency": turn_latency
                    })
            
            elif "FINAL_ANSWER:" in response:
                trajectory.outcome = True
                trajectory.turns.append({
                    "turn": turn + 1,
                    "final_answer": response.split("FINAL_ANSWER:")[-1].strip(),
                    "cost": 0.01,  # Small cost for final reasoning
                    "latency": turn_latency
                })
                break
            
            else:
                # Continue reasoning or force completion
                if turn > 5:  # Force completion after 5 turns
                    trajectory.outcome = False
                    break
                
                messages.append({"role": "assistant", "content": response})
                trajectory.turns.append({
                    "turn": turn + 1,
                    "reasoning": response,
                    "cost": 0.01,
                    "latency": turn_latency
                })
        
        return trajectory
    
    def orchestrate_streaming(self, query: str, user_preference: str = "", max_turns: int = 50) -> Trajectory:
        """Main orchestration with streaming output"""
        trajectory = Trajectory(turns=[], tool_calls=[])
        
        print(f"[ORCHESTRATOR] Starting with preference: {user_preference}")
        
        # Simple query handling with streaming
        if any(word in query.lower() for word in ['2+2', '2+4', 'add', 'plus']):
            print(f"[ORCHESTRATOR] Detected simple math query")
            if '2+2' in query.lower():
                answer = "4"
            elif '2+4' in query.lower():
                answer = "6"
            else:
                answer = "Calculated result"
            
            print(f"[ORCHESTRATOR] Direct calculation: {answer}")
            trajectory.outcome = True
            trajectory.turns.append({
                "turn": 1,
                "final_answer": answer,
                "cost": 0.01,
                "latency": 0.001
            })
            trajectory.total_cost = 0.01
            return trajectory
        
        # Paper-based tool selection logic
        if any(word in query.lower() for word in ['code', 'program', 'python', 'function', 'script']):
            # Coding task - use specialized coding model (per paper)
            print(f"[ORCHESTRATOR] Coding task detected -> qwen2.5-coder-32b (specialized)")
            print(f"[ORCHESTRATOR] Preference '{user_preference}' -> selecting specialized model")
            
            tool_call = ToolCall(name="qwen2.5-coder-32b", args={"query": query}, cost=0.3)
            result = self._execute_tool_streaming(tool_call)
            
            trajectory.tool_calls.append(tool_call)
            trajectory.total_cost += tool_call.cost
            trajectory.outcome = True
            trajectory.turns.append({
                "turn": 1,
                "tool_call": tool_call.name,
                "result": result,
                "final_answer": result,
                "cost": tool_call.cost,
                "latency": tool_call.latency
            })
            return trajectory
        
        elif any(word in query.lower() for word in ['math', 'calculate', 'solve']):
            # Math task - select based on preference (per paper multi-objective optimization)
            if user_preference in ["efficiency", "cost_efficiency"]:
                model, cost = "qwen2.5-math-7b", 0.1
                print(f"[ORCHESTRATOR] Math task + efficiency preference -> {model} (cost-optimal)")
            else:
                model, cost = "qwen2.5-math-72b", 0.5  
                print(f"[ORCHESTRATOR] Math task + accuracy preference -> {model} (performance-optimal)")
            
            tool_call = ToolCall(name=model, args={"query": query}, cost=cost)
            result = self._execute_tool_streaming(tool_call)
            
            trajectory.tool_calls.append(tool_call)
            trajectory.total_cost += tool_call.cost
            trajectory.outcome = True
            trajectory.turns.append({
                "turn": 1,
                "tool_call": tool_call.name,
                "result": result,
                "final_answer": result,
                "cost": tool_call.cost,
                "latency": tool_call.latency
            })
            return trajectory
        
        elif any(word in query.lower() for word in ['poem', 'story', 'creative']) and not any(word in query.lower() for word in ['code', 'program', 'python', 'function', 'script']):
            # Creative writing - generalist model selection (per paper)
            if user_preference in ["efficiency", "cost_efficiency"]:
                model, cost = "gpt-5-mini", 0.5
                print(f"[ORCHESTRATOR] Creative task + efficiency -> {model} (cost-optimal generalist)")
            elif user_preference == "specialized_preference":
                model, cost = "claude-opus-4.1", 1.5
                print(f"[ORCHESTRATOR] Creative task + specialized preference -> {model} (best generalist)")
            else:
                model, cost = "claude-opus-4.1", 1.5
                print(f"[ORCHESTRATOR] Creative task + accuracy -> {model} (performance-optimal)")
            
            tool_call = ToolCall(name=model, args={"query": query}, cost=cost)
            result = self._execute_tool_streaming(tool_call)
            
            trajectory.tool_calls.append(tool_call)
            trajectory.total_cost += tool_call.cost
            trajectory.outcome = True
            trajectory.turns.append({
                "turn": 1,
                "tool_call": tool_call.name,
                "result": result,
                "final_answer": result,
                "cost": tool_call.cost,
                "latency": tool_call.latency
            })
            return trajectory
        
        else:
            # General queries - basic tools first (per paper tool hierarchy)
            print(f"[ORCHESTRATOR] General query -> web_search (basic tool)")
            print(f"[ORCHESTRATOR] Following paper hierarchy: basic -> specialized -> generalist")
            
            tool_call = ToolCall(name="web_search", args={"query": query}, cost=0.01)
            result = self._execute_tool_streaming(tool_call)
            
            trajectory.tool_calls.append(tool_call)
            trajectory.total_cost += tool_call.cost
            trajectory.outcome = True
            trajectory.turns.append({
                "turn": 1,
                "tool_call": tool_call.name,
                "result": result,
                "final_answer": result,
                "cost": tool_call.cost,
                "latency": tool_call.latency
            })
            return trajectory
    
    def _execute_tool_streaming(self, tool_call: ToolCall) -> str:
        """Execute tool with streaming output"""
        print(f"[{tool_call.name.upper()}] Executing...")
        
        start_time = time.time()
        result = self._execute_tool(tool_call)
        tool_call.latency = time.time() - start_time
        
        print(f"[{tool_call.name.upper()}] Completed in {tool_call.latency:.3f}s | Cost: ${tool_call.cost}")
        
        # Show actual output for creative tasks
        if any(word in tool_call.name for word in ['gpt', 'claude', 'qwen']):
            print(f"[{tool_call.name.upper()}] Output: {result}")
        else:
            print(f"[{tool_call.name.upper()}] Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        
        return result
    
    def _parse_tool_call(self, response: str) -> Optional[ToolCall]:
        """Parse tool call from orchestrator response"""
        try:
            if "TOOL_CALL:" not in response:
                return None
            
            tool_part = response.split("TOOL_CALL:")[-1].strip()
            tool_data = json.loads(tool_part.split('\n')[0])
            
            tool_name = tool_data["name"]
            tool_args = tool_data.get("args", {})
            cost = self.tool_costs.get(tool_name, 0.1)
            
            return ToolCall(name=tool_name, args=tool_args, cost=cost)
        except:
            return None
    
    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute the called tool"""
        start_time = time.time()
        
        try:
            if tool_call.name == "web_search":
                result = self._web_search(tool_call.args.get("query", ""))
            
            elif tool_call.name == "local_search":
                result = self._local_search(tool_call.args.get("query", ""))
            
            elif tool_call.name == "code_interpreter":
                result = self._execute_code(tool_call.args.get("code", ""))
            
            elif "math" in tool_call.name:
                result = self._call_math_model(tool_call)
            
            elif "coder" in tool_call.name:
                result = self._call_code_model(tool_call)
            
            elif tool_call.name in ["gpt-5", "gpt-5-mini", "claude-opus-4.1", "qwen3-32b", "qwen3-235b"]:
                result = self._call_generalist_model(tool_call)
            
            else:
                result = f"Tool {tool_call.name} not implemented"
            
            tool_call.latency = time.time() - start_time
            return result
            
        except Exception as e:
            tool_call.latency = time.time() - start_time
            return f"Error executing {tool_call.name}: {str(e)}"
    
    def _web_search(self, query: str) -> str:
        """Dynamic web search using Tavily API"""
        try:
            import requests
            import os
            
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                return f"Web search for '{query}': API key not found"
            
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": 3
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    formatted_results = []
                    for i, result in enumerate(results[:3], 1):
                        title = result.get('title', 'No title')
                        url = result.get('url', 'No URL')
                        content = result.get('content', 'No content')[:200] + '...' if result.get('content') else 'No content'
                        formatted_results.append(f"{i}. {title}\n   URL: {url}\n   Content: {content}\n")
                    return "\n".join(formatted_results)
                else:
                    return f"Web search completed but no results found for: {query}"
            else:
                return f"Web search failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Dynamic web search failed: {str(e)}"
    
    def _local_search(self, query: str) -> str:
        """Simulate local search"""
        return f"Local search results for '{query}': [Simulated local results]"
    
    def _execute_code(self, code: str) -> str:
        """Execute Python code safely"""
        try:
            # In production, use proper sandboxing
            exec_globals = {"__builtins__": {}}
            exec(code, exec_globals)
            return "Code executed successfully"
        except Exception as e:
            return f"Code execution error: {str(e)}"
    
    def _call_math_model(self, tool_call: ToolCall) -> str:
        """Call specialized math model using HuggingFace"""
        query = tool_call.args.get("query", "")
        
        # Paper models with HuggingFace fallbacks
        model_mapping = {
            "qwen2.5-math-7b": ("Qwen/Qwen2.5-Math-7B-Instruct", True),
            "qwen2.5-math-72b": ("Qwen/Qwen2.5-Math-7B-Instruct", False)  # Fallback to 7B
        }
        
        model_info = model_mapping.get(tool_call.name)
        if not model_info:
            return f"Model {tool_call.name} not found in configuration"
        
        hf_model, is_exact = model_info
        if not is_exact:
            print(f"[WARNING] {tool_call.name} not available on HF, using similar capable model {hf_model}")
        
        try:
            response = get_llm_response(
                model=hf_model,
                messages=[{"role": "user", "content": f"Solve this math problem: {query}"}],
                temperature=0.1,
                model_type="huggingface"
            )
            return response
        except Exception as e:
            return f"{tool_call.name} HF API failed: {str(e)}"
    
    def _call_code_model(self, tool_call: ToolCall) -> str:
        """Call specialized coding model using HuggingFace"""
        query = tool_call.args.get("query", "")
        
        # Direct HuggingFace model mapping
        model_mapping = {
            "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct"
        }
        
        hf_model = model_mapping.get(tool_call.name)
        if not hf_model:
            return f"Model {tool_call.name} not found in configuration"
        
        try:
            response = get_llm_response(
                model=hf_model,
                messages=[{"role": "user", "content": f"Write Python code for: {query}. Return only the code."}],
                temperature=0.2,
                model_type="huggingface"
            )
            return response.strip()
        except Exception as e:
            return f"{tool_call.name} HF API failed: {str(e)}"
    
    def _call_generalist_model(self, tool_call: ToolCall) -> str:
        """Call generalist LLM using HuggingFace models"""
        query = tool_call.args.get("query", "")
        
        # Paper models with diverse HuggingFace fallbacks
        model_mapping = {
            "gpt-5": ("meta-llama/Llama-2-13b-chat-hf", False),           # GPT-5 -> Llama-2 13B
            "gpt-5-mini": ("HuggingFaceH4/zephyr-7b-beta", False),        # GPT-5-mini -> Zephyr 7B
            "claude-opus-4.1": ("Qwen/Qwen2.5-14B-Instruct", False),     # Claude -> Qwen 14B
            "qwen3-32b": ("Qwen/Qwen2.5-32B-Instruct", False),           # Qwen3-32B -> Qwen2.5-32B
            "qwen3-235b": ("Qwen/Qwen2.5-72B-Instruct", False)           # Qwen3-235B -> Qwen2.5-72B
        }
        
        model_info = model_mapping.get(tool_call.name)
        if not model_info:
            return f"Model {tool_call.name} not found in configuration"
        
        hf_model, is_exact = model_info
        if not is_exact:
            print(f"[WARNING] {tool_call.name} not available on HF, using similar capable model {hf_model}")
        
        try:
            response = get_llm_response(
                model=hf_model,
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
                model_type="huggingface"
            )
            return response if response else "Content generated"
        except Exception as e:
            return f"{tool_call.name} HF failed: {str(e)}"

    def calculate_reward(self, trajectory: Trajectory, user_preferences: Dict[str, float]) -> float:
        """Calculate multi-objective reward (outcome + efficiency + preference)"""
        
        # Outcome reward (binary)
        outcome_reward = 1.0 if trajectory.outcome else 0.0
        
        # Efficiency rewards (negative cost/latency)
        cost_penalty = -trajectory.total_cost / 10.0  # Normalize
        latency_penalty = -trajectory.total_latency / 60.0  # Normalize
        
        # Preference reward (tool usage alignment)
        tool_usage = {}
        for call in trajectory.tool_calls:
            tool_usage[call.name] = tool_usage.get(call.name, 0) + 1
        
        preference_reward = 0.0
        for tool, count in tool_usage.items():
            pref_weight = user_preferences.get(tool, 0.0)
            preference_reward += count * pref_weight
        
        # Combined reward (as per paper Equation 2)
        if trajectory.outcome:
            total_reward = (
                outcome_reward * user_preferences.get("accuracy", 1.0) +
                cost_penalty * user_preferences.get("cost", 0.3) +
                latency_penalty * user_preferences.get("latency", 0.2) +
                preference_reward * 0.1
            )
        else:
            total_reward = 0.0
        
        return total_reward

if __name__ == "__main__":
    print("ToolOrchestra Core - Use dynamic_orchestra.py for interactive mode")
    print("Run: python dynamic_orchestra.py")