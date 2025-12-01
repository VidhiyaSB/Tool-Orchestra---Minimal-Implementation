# ToolOrchestra - Exact Paper Implementation

**Elevating Intelligence via Efficient Model and Tool Orchestration**

This is the exact implementation of ToolOrchestra as described in the NVIDIA research paper (2025).

## Key Features

### 🎯 Core Architecture
- **8B Orchestrator Model** - Small model that coordinates intelligent tools
- **Multi-turn Reasoning** - Reasoning → Action → Observation loops
- **Intelligent Tool Calling** - Uses other LLMs as tools, not just utilities

### 🛠️ Tool Categories

**Basic Tools:**
- Web Search (Tavily API)
- Local Search (Faiss + Qwen3-Embedding-8B) 
- Code Interpreter (Python sandbox)

**Specialized LLMs:**
- Qwen2.5-Math-7B/72B (Mathematical reasoning)
- Qwen2.5-Coder-32B (Code generation)

**Generalist LLMs:**
- GPT-5, GPT-5-mini
- Claude Opus 4.1
- Qwen3-32B, Qwen3-235B

### 🎯 Reinforcement Learning
- **GRPO Algorithm** - Group Relative Policy Optimization
- **Multi-objective Rewards:**
  - Outcome correctness
  - Cost efficiency  
  - Latency optimization
  - User preference alignment

### 📊 Results (Paper Benchmarks)
- **HLE**: 37.1% (vs GPT-5: 35.1%) at 2.5x efficiency
- **FRAMES**: 76.3% (vs GPT-5: 74.0%) at 30% cost
- **τ2-Bench**: 80.2% (vs GPT-5: 77.7%) at 30% cost

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
set TAVILY_KEY=your_tavily_api_key
set CLIENT_ID=your_client_id
set CLIENT_SECRET=your_client_secret
set OSS_KEY=your_oss_key
```

3. **Run the orchestrator:**
```bash
python run_tool_orchestra.py
```

## File Structure

```
├── tool_orchestra.py          # Main orchestrator implementation
├── reinforcement_learning.py  # GRPO training & data synthesis
├── paper_tools.json          # Tool configurations from paper
├── run_tool_orchestra.py     # Complete evaluation suite
├── LLM_CALL.py              # LLM service interface
└── README_PAPER.md          # This file
```

## Usage Examples

### Basic Orchestration
```python
from tool_orchestra import ToolOrchestra

orchestrator = ToolOrchestra()
trajectory = orchestrator.orchestrate(
    query="What is the Alon-Tarsi number of K_{1000,1000}?",
    user_preference="I want to be cost efficient if possible"
)

print(f"Cost: ${trajectory.total_cost:.3f}")
print(f"Success: {trajectory.outcome}")
```

### Preference-Aware Orchestration
```python
# Cost-efficient preference
trajectory1 = orchestrator.orchestrate(
    query="Solve this differential equation: dy/dx = x²y",
    user_preference="Minimize cost while maintaining accuracy"
)

# Accuracy-focused preference  
trajectory2 = orchestrator.orchestrate(
    query="Solve this differential equation: dy/dx = x²y", 
    user_preference="Prioritize accuracy over speed"
)
```

### Reinforcement Learning Training
```python
from reinforcement_learning import GroupRelativePolicyOptimization, ToolScaleDataSynthesis

# Generate training data
data_gen = ToolScaleDataSynthesis()
training_data = data_gen.generate_training_data()

# Train with GRPO
grpo = GroupRelativePolicyOptimization(orchestrator_model)
grpo.update_policy(batch)
```

## Key Innovations

1. **Orchestration Paradigm**: Small model coordinates larger models
2. **Multi-objective RL**: Balances accuracy, cost, latency, preferences
3. **Intelligent Tools**: Treats LLMs as tools with different capabilities
4. **Strategic Routing**: Math → Math models, Code → Code models, etc.
5. **Cost Awareness**: Optimizes for efficiency while maintaining performance

## Architecture Details

### Multi-turn Process
```
1. REASONING: Analyze current state and plan next action
2. TOOL CALLING: Select and invoke appropriate tool
3. OBSERVATION: Process tool response and continue
4. Repeat until task completion or max turns
```

### Reward Function
```python
reward = outcome_reward * w_accuracy + 
         cost_penalty * w_cost + 
         latency_penalty * w_latency + 
         preference_alignment * w_preference
```

### Tool Selection Strategy
- **Math problems** → Qwen2.5-Math-7B/72B
- **Coding tasks** → Qwen2.5-Coder-32B  
- **Complex reasoning** → GPT-5, Claude Opus
- **Simple queries** → GPT-5-mini, local search
- **Web info** → Tavily search

## Performance Comparison

| Model | HLE | FRAMES | τ2-Bench | Cost | Efficiency |
|-------|-----|--------|----------|------|------------|
| GPT-5 | 35.1% | 74.0% | 77.7% | $0.30 | 1.0x |
| Claude Opus 4.1 | 34.6% | 72.8% | 76.8% | $0.53 | 0.7x |
| **Orchestrator-8B** | **37.1%** | **76.3%** | **80.2%** | **$0.09** | **2.5x** |

## Citation

```bibtex
@article{su2025toolorchestra,
  title={ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration},
  author={Su, Hongjin and Diao, Shizhe and Lu, Ximing and others},
  journal={arXiv preprint arXiv:2511.21689},
  year={2025}
}
```

## License

Apache 2.0 - See paper for full details.

---

*This implementation follows the exact methodology described in the NVIDIA ToolOrchestra paper (2025).*