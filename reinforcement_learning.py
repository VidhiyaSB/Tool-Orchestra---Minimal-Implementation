import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any
from dataclasses import dataclass
from tool_orchestra import ToolOrchestra, Trajectory

@dataclass
class RLBatch:
    trajectories: List[Trajectory]
    rewards: List[float]
    advantages: List[float]

class GroupRelativePolicyOptimization:
    """GRPO implementation for ToolOrchestra training"""
    
    def __init__(self, orchestrator_model, learning_rate=1e-6, clip_epsilon=0.2):
        self.orchestrator = orchestrator_model
        self.lr = learning_rate
        self.clip_epsilon = clip_epsilon
        self.optimizer = torch.optim.Adam(orchestrator_model.parameters(), lr=learning_rate)
    
    def compute_rewards(self, trajectories: List[Trajectory], user_preferences: Dict[str, float]) -> List[float]:
        """Compute multi-objective rewards for trajectories"""
        rewards = []
        
        for trajectory in trajectories:
            # Outcome reward (binary)
            outcome_reward = 1.0 if trajectory.outcome else 0.0
            
            # Efficiency rewards (normalized)
            cost_reward = -trajectory.total_cost / 10.0
            latency_reward = -trajectory.total_latency / 60.0
            
            # Preference reward (tool usage alignment)
            tool_counts = {}
            for call in trajectory.tool_calls:
                tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
            
            preference_reward = 0.0
            for tool, count in tool_counts.items():
                pref_weight = user_preferences.get(tool, 0.0)
                preference_reward += count * pref_weight
            
            # Combined reward (Equation 2 from paper)
            if trajectory.outcome:
                total_reward = (
                    outcome_reward * user_preferences.get("accuracy", 1.0) +
                    cost_reward * user_preferences.get("cost_efficiency", 0.3) +
                    latency_reward * user_preferences.get("latency_efficiency", 0.2) +
                    preference_reward * 0.1
                )
            else:
                total_reward = 0.0
            
            rewards.append(total_reward)
        
        return rewards
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute advantages using group normalization (Equation 3)"""
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        if std_reward == 0:
            return [0.0] * len(rewards)
        
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        return advantages
    
    def update_policy(self, batch: RLBatch):
        """Update orchestrator policy using GRPO"""
        
        # Filter trajectories (homogeneity, format, validity)
        filtered_batch = self._filter_batch(batch)
        
        if len(filtered_batch.trajectories) == 0:
            return
        
        # Compute policy loss
        policy_loss = 0.0
        
        for trajectory, advantage in zip(filtered_batch.trajectories, filtered_batch.advantages):
            # Get log probabilities for actions in trajectory
            log_probs = self._get_trajectory_log_probs(trajectory)
            
            # Compute likelihood ratio (simplified)
            ratio = torch.exp(log_probs)  # Assuming old_log_probs = 0 for simplicity
            
            # Clipped surrogate objective (Equation 4)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss += -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        policy_loss = policy_loss / len(filtered_batch.trajectories)
        
        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
    
    def _filter_batch(self, batch: RLBatch) -> RLBatch:
        """Apply filtering as described in paper"""
        filtered_trajectories = []
        filtered_rewards = []
        filtered_advantages = []
        
        # Homogeneity filtering
        reward_std = np.std(batch.rewards)
        if reward_std < 0.1:
            return RLBatch([], [], [])
        
        for i, (traj, reward, advantage) in enumerate(zip(batch.trajectories, batch.rewards, batch.advantages)):
            # Format consistency filtering
            if not self._is_valid_format(traj):
                continue
            
            # Invalid output filtering  
            if not traj.outcome and len(traj.turns) < 2:
                continue
            
            filtered_trajectories.append(traj)
            filtered_rewards.append(reward)
            filtered_advantages.append(advantage)
        
        return RLBatch(filtered_trajectories, filtered_rewards, filtered_advantages)
    
    def _is_valid_format(self, trajectory: Trajectory) -> bool:
        """Check if trajectory has valid tool call format"""
        for turn in trajectory.turns:
            if 'tool_call' in turn and not turn['tool_call']:
                return False
        return True
    
    def _get_trajectory_log_probs(self, trajectory: Trajectory) -> torch.Tensor:
        """Get log probabilities for trajectory actions (simplified)"""
        # In real implementation, this would compute actual log probabilities
        # from the orchestrator model for each action in the trajectory
        return torch.tensor(0.0, requires_grad=True)

class ToolScaleDataSynthesis:
    """Data synthesis pipeline for ToolScale dataset"""
    
    def __init__(self):
        self.domains = [
            "finance", "sport", "e-commerce", "medicine", "entertainment",
            "railway", "restaurant", "education", "travel", "weather"
        ]
    
    def generate_training_data(self, num_examples_per_domain=100) -> List[Dict[str, Any]]:
        """Generate synthetic training data across domains"""
        training_data = []
        
        for domain in self.domains:
            # Generate domain-specific database schema
            schema = self._generate_schema(domain)
            
            # Generate tools for domain
            tools = self._generate_domain_tools(domain)
            
            # Generate tasks with golden actions
            for _ in range(num_examples_per_domain):
                task = self._generate_task(domain, schema, tools)
                training_data.append(task)
        
        return training_data
    
    def _generate_schema(self, domain: str) -> Dict[str, Any]:
        """Generate database schema for domain"""
        # Simplified schema generation
        return {
            "domain": domain,
            "tables": [f"{domain}_table"],
            "fields": ["id", "name", "status", "value"]
        }
    
    def _generate_domain_tools(self, domain: str) -> List[Dict[str, Any]]:
        """Generate domain-specific tools"""
        return [
            {"name": f"{domain}_query", "type": "database"},
            {"name": f"{domain}_update", "type": "database"},
            {"name": f"{domain}_search", "type": "search"}
        ]
    
    def _generate_task(self, domain: str, schema: Dict, tools: List[Dict]) -> Dict[str, Any]:
        """Generate a task with golden actions"""
        return {
            "domain": domain,
            "task": f"Complete {domain} related task",
            "golden_actions": [f"call_{domain}_query", f"call_{domain}_update"],
            "eval_criteria": ["execution_correctness", "process_fidelity", "operation_completeness"]
        }

class PreferenceGenerator:
    """Generate user preference pairs for training"""
    
    def generate_preference_pairs(self, tools: List[str], num_pairs=1000) -> List[Dict[str, Any]]:
        """Generate (instruction, preference_vector) pairs"""
        pairs = []
        
        for _ in range(num_pairs):
            # Generate preference instruction
            instruction = self._generate_preference_instruction()
            
            # Generate corresponding preference vector
            preference_vector = self._generate_preference_vector(tools, instruction)
            
            pairs.append({
                "instruction": instruction,
                "preference_vector": preference_vector
            })
        
        return pairs
    
    def _generate_preference_instruction(self) -> str:
        """Generate natural language preference instruction"""
        preferences = [
            "I want to be cost efficient if possible",
            "Please prioritize accuracy over speed",
            "Use local models when available for privacy",
            "Minimize latency for real-time applications",
            "Prefer specialized models for domain tasks"
        ]
        return np.random.choice(preferences)
    
    def _generate_preference_vector(self, tools: List[str], instruction: str) -> List[float]:
        """Generate preference vector based on instruction"""
        vector = [0.0] * (len(tools) + 3)  # tools + accuracy + cost + latency
        
        if "cost efficient" in instruction:
            vector[-2] = 1.0  # High cost preference
        elif "accuracy" in instruction:
            vector[-3] = 1.0  # High accuracy preference  
        elif "local" in instruction:
            # Prefer local tools
            for i, tool in enumerate(tools):
                if "local" in tool:
                    vector[i] = 1.0
        
        return vector

if __name__ == "__main__":
    # Example usage
    print("ToolOrchestra Reinforcement Learning Components")
    
    # Generate synthetic data
    data_generator = ToolScaleDataSynthesis()
    training_data = data_generator.generate_training_data(num_examples_per_domain=10)
    print(f"Generated {len(training_data)} training examples")
    
    # Generate preference pairs
    pref_generator = PreferenceGenerator()
    tools = ["web_search", "local_search", "gpt-5", "math_model"]
    pref_pairs = pref_generator.generate_preference_pairs(tools, num_pairs=10)
    print(f"Generated {len(pref_pairs)} preference pairs")
    
    for pair in pref_pairs[:3]:
        print(f"Instruction: {pair['instruction']}")
        print(f"Vector: {pair['preference_vector']}")
        print()