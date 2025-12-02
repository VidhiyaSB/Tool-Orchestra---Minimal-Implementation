
# 🎼 Tool Orchestra - Proof of Concept

> **⚡ PROOF OF CONCEPT: Intelligent AI Task Routing**

Functional implementation inspired by NVIDIA's ToolOrchestra paper. Demonstrates preference-aware routing using NVIDIA Orchestrator-8B with HuggingFace models.


## 🎯 **What's Actually Working**

✅ **NVIDIA Orchestrator-8B** - Real AI routing decisions via Ollama  
✅ **Preference-Aware Routing** - Cost vs accuracy tradeoffs  
✅ **HuggingFace Model Execution** - Math, coding, creative tasks  
✅ **Web Search** - Real Tavily API integration    
✅ **Cost Tracking** - Per-tool cost calculation  

## 🔄 **Paper Model → HuggingFace Equivalent**

**Specialized Models (Exact Match):**
- `qwen2.5-math-7b` → `Qwen/Qwen2.5-Math-7B-Instruct` ✅
- `qwen2.5-coder-32b` → `Qwen/Qwen2.5-Coder-32B-Instruct` ✅

**Generalist Models (Equivalents):**
- `gpt-5` → `meta-llama/Llama-2-13b-chat-hf`
- `gpt-5-mini` → `HuggingFaceH4/zephyr-7b-beta`
- `claude-opus-4.1` → `Qwen/Qwen2.5-14B-Instruct`
- `qwen3-32b` → `Qwen/Qwen2.5-32B-Instruct`
- `qwen3-235b` → `Qwen/Qwen2.5-72B-Instruct`

## 🚀 **Setup**

```bash
# Install Ollama + NVIDIA model
winget install Ollama.Ollama
ollama pull hf.co/bartowski/nvidia_Orchestrator-8B-GGUF:IQ2_M

# Setup Python environment
pip install -r requirements.txt

# Add API keys to .env
HF_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_key

# Run
python dynamic_orchestra.py
```

## 📊 **Preference-Based Routing Examples**

### **Math Tasks:**
```bash
# Cost efficiency → cheaper math model
"Solve x² + 5x + 6 = 0" + cost_efficiency → qwen2.5-math-7b ($0.10)

# Accuracy → better math model  
"Solve x² + 5x + 6 = 0" + accuracy → qwen2.5-math-72b ($0.50)
```

### **Creative Writing:**
```bash
# Cost efficiency → lighter creative model
"Write a poem about AI" + cost_efficiency → gpt-5-mini ($0.50)

# Accuracy → most capable creative model
"Write a poem about AI" + accuracy → qwen3-235b ($1.20)
```

### **Coding Tasks:**
```bash
# All preferences use coding specialist
"Write Python sorting code" → qwen2.5-coder-32b ($0.30)
```

## 🎭 **How It Works**

### **1. NVIDIA Orchestrator-8B Decision Making:**
- **Analyzes task type** (math, creative, coding, general)
- **Considers user preference** (cost vs accuracy vs speed)
- **Selects optimal tool** from available options
- **Provides reasoning** for the decision

### **2. Model Execution:**
- **Specialist models** handle domain-specific tasks
- **Generalist models** handle creative/general queries
- **Basic tools** handle search and code execution

### **3. Cost Optimization:**
- **Tracks per-tool costs** based on paper pricing
- **Shows cost-benefit analysis** for each decision
- **Enables cost-conscious routing** when requested

## ❌ **What's Missing from Paper**

- **No GRPO Training** - Uses pre-trained NVIDIA model
- **No ToolScale Dataset** - No synthetic training data
- **No Multi-Objective RL** - No learned cost optimization
- **No Benchmarking** - No HLE/FRAMES/τ2-Bench evaluation

## 🔗 **References**

- **Paper**: [ToolOrchestra Research](https://arxiv.org/abs/2511.21689)
- **NVIDIA Model**: [Orchestrator-8B GGUF](https://huggingface.co/bartowski/nvidia_Orchestrator-8B-GGUF)
official - https://huggingface.co/nvidia/Orchestrator-8B

---

**This demonstrates the core orchestration concept with real NVIDIA routing + HuggingFace execution.**
