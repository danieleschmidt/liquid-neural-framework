import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from collections import deque


class ContinuousLearner:
    """
    Continuous learning framework for liquid neural networks.
    
    Implements online learning, continual adaptation, and catastrophic
    forgetting mitigation for liquid neural network architectures.
    """
    
    def __init__(
        self,
        model,
        memory_size: int = 1000,
        adaptation_rate: float = 0.01,
        consolidation_strength: float = 0.1,
        plasticity_decay: float = 0.99
    ):
        self.model = model
        self.memory_size = memory_size
        self.adaptation_rate = adaptation_rate
        self.consolidation_strength = consolidation_strength
        self.plasticity_decay = plasticity_decay
        
        # Experience replay buffer
        self.memory_buffer = deque(maxlen=memory_size)
        
        # Importance weights for parameters (Fisher Information approximation)
        self.importance_weights = {}
        
        # Consolidated parameters (anchor points)
        self.consolidated_params = {}
        
        # Plasticity modulation
        self.plasticity = {}
        
        # Adaptation history
        self.adaptation_history = []
        
        self._initialize_continual_learning_state()
    
    def _initialize_continual_learning_state(self):
        """Initialize state for continual learning."""
        for key, param in self.model.params.items():
            self.importance_weights[key] = jnp.ones_like(param)
            self.consolidated_params[key] = param.copy()
            self.plasticity[key] = jnp.ones_like(param)
    
    def add_experience(
        self, 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add new experience to memory buffer."""
        experience = {
            'inputs': inputs,
            'targets': targets,
            'timestamp': len(self.memory_buffer),
            'metadata': metadata or {}
        }
        self.memory_buffer.append(experience)
    
    def sample_experiences(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample experiences from memory buffer."""
        if len(self.memory_buffer) < batch_size:
            return list(self.memory_buffer)
        
        indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        return [self.memory_buffer[i] for i in indices]
    
    def compute_fisher_information(
        self,
        experiences: List[Dict[str, Any]],
        n_samples: int = 100
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute Fisher Information Matrix diagonal approximation.
        
        Used for identifying important parameters that should be preserved
        during continual learning.
        """
        fisher_info = {key: jnp.zeros_like(param) for key, param in self.model.params.items()}
        
        # Sample from experiences
        sampled_experiences = experiences[:min(n_samples, len(experiences))]
        
        for experience in sampled_experiences:
            inputs = experience['inputs']
            targets = experience['targets']
            
            # Compute gradients of log-likelihood
            def log_likelihood(params):
                # Update model with current params
                old_params = self.model.params.copy()
                self.model.update_params(params)
                
                # Forward pass
                outputs, _ = self.model.forward(inputs)
                
                # Compute log likelihood (assuming Gaussian)
                log_prob = -0.5 * jnp.sum((outputs - targets) ** 2)
                
                # Restore original params
                self.model.update_params(old_params)
                
                return log_prob
            
            # Compute gradients
            grads = jax.grad(log_likelihood)(self.model.params)
            
            # Accumulate squared gradients (Fisher Information)
            for key in fisher_info:
                fisher_info[key] += grads[key] ** 2
        
        # Normalize by number of samples
        n_samples_actual = len(sampled_experiences)
        for key in fisher_info:
            fisher_info[key] /= n_samples_actual
        
        return fisher_info
    
    def update_importance_weights(self, new_experiences: List[Dict[str, Any]]):
        """Update importance weights based on new experiences."""
        if not new_experiences:
            return
        
        # Compute Fisher information for new experiences
        new_fisher = self.compute_fisher_information(new_experiences)
        
        # Update importance weights with exponential moving average
        alpha = 0.1  # Update rate
        for key in self.importance_weights:
            self.importance_weights[key] = (
                (1 - alpha) * self.importance_weights[key] + 
                alpha * new_fisher[key]
            )
    
    def consolidation_loss(
        self, 
        current_params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Compute consolidation loss to prevent catastrophic forgetting.
        
        Uses Fisher Information to weight the importance of preserving
        different parameters.
        """
        total_loss = 0.0
        
        for key in current_params:
            if key in self.consolidated_params:
                param_diff = current_params[key] - self.consolidated_params[key]
                importance = self.importance_weights[key]
                consolidation_term = 0.5 * importance * (param_diff ** 2)
                total_loss += jnp.sum(consolidation_term)
        
        return self.consolidation_strength * total_loss
    
    def adaptive_plasticity_loss(
        self,
        current_params: Dict[str, jnp.ndarray],
        prediction_loss: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute adaptive plasticity loss that modulates learning rate
        based on parameter importance and current performance.
        """
        plasticity_loss = 0.0
        
        for key in current_params:
            if key in self.plasticity:
                # Plasticity modulates how much each parameter can change
                param_magnitude = jnp.sum(current_params[key] ** 2)
                plasticity_constraint = (1.0 - self.plasticity[key]) * param_magnitude
                plasticity_loss += jnp.sum(plasticity_constraint)
        
        # Scale by prediction performance (worse performance allows more plasticity)
        performance_factor = jnp.tanh(prediction_loss)
        
        return 0.01 * performance_factor * plasticity_loss
    
    def continual_learning_step(
        self,
        new_inputs: jnp.ndarray,
        new_targets: jnp.ndarray,
        replay_batch_size: int = 32,
        update_consolidation: bool = False
    ) -> Dict[str, float]:
        """
        Single step of continual learning with experience replay and consolidation.
        """
        # Add new experience
        self.add_experience(new_inputs, new_targets)
        
        # Sample from experience replay buffer
        replay_experiences = self.sample_experiences(replay_batch_size)
        
        # Prepare training data (mix of new and replayed experiences)
        all_inputs = [new_inputs]
        all_targets = [new_targets]
        
        for exp in replay_experiences:
            all_inputs.append(exp['inputs'])
            all_targets.append(exp['targets'])
        
        # Compute standard prediction loss
        total_pred_loss = 0.0
        for inputs, targets in zip(all_inputs, all_targets):
            outputs, _ = self.model.forward(inputs)
            pred_loss = jnp.mean((outputs - targets) ** 2)
            total_pred_loss += pred_loss
        
        total_pred_loss /= len(all_inputs)
        
        # Compute consolidation loss
        consol_loss = self.consolidation_loss(self.model.params)
        
        # Compute adaptive plasticity loss
        plasticity_loss = self.adaptive_plasticity_loss(self.model.params, total_pred_loss)
        
        # Total loss
        total_loss = total_pred_loss + consol_loss + plasticity_loss
        
        # Update model parameters (simplified gradient step)
        def loss_fn(params):
            old_params = self.model.params.copy()
            self.model.update_params(params)
            
            pred_loss = 0.0
            for inputs, targets in zip(all_inputs, all_targets):
                outputs, _ = self.model.forward(inputs)
                pred_loss += jnp.mean((outputs - targets) ** 2)
            
            pred_loss /= len(all_inputs)
            consol = self.consolidation_loss(params)
            plast = self.adaptive_plasticity_loss(params, pred_loss)
            
            self.model.update_params(old_params)
            return pred_loss + consol + plast
        
        # Compute gradients and update
        grads = jax.grad(loss_fn)(self.model.params)
        
        # Apply gradients with adaptive learning rate
        new_params = {}
        for key, param in self.model.params.items():
            adaptive_lr = self.adaptation_rate * self.plasticity.get(key, 1.0)
            new_params[key] = param - adaptive_lr * grads[key]
        
        self.model.update_params(new_params)
        
        # Update plasticity (decay over time)
        for key in self.plasticity:
            self.plasticity[key] *= self.plasticity_decay
        
        # Update importance weights periodically
        if len(self.memory_buffer) % 100 == 0:
            recent_experiences = list(self.memory_buffer)[-100:]
            self.update_importance_weights(recent_experiences)
        
        # Consolidation step
        if update_consolidation:
            self.consolidate_knowledge()
        
        # Record adaptation metrics
        metrics = {
            'prediction_loss': float(total_pred_loss),
            'consolidation_loss': float(consol_loss),
            'plasticity_loss': float(plasticity_loss),
            'total_loss': float(total_loss),
            'memory_size': len(self.memory_buffer),
            'avg_plasticity': float(jnp.mean(jnp.concatenate([
                p.flatten() for p in self.plasticity.values()
            ])))
        }
        
        self.adaptation_history.append(metrics)
        
        return metrics
    
    def consolidate_knowledge(self):
        """
        Consolidate current knowledge by updating anchor parameters
        and resetting plasticity.
        """
        # Update consolidated parameters to current parameters
        for key, param in self.model.params.items():
            self.consolidated_params[key] = param.copy()
        
        # Reset plasticity to allow new learning
        for key in self.plasticity:
            self.plasticity[key] = jnp.ones_like(self.plasticity[key])
        
        print(f"Knowledge consolidated at step {len(self.adaptation_history)}")
    
    def evaluate_forgetting(
        self,
        old_experiences: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate catastrophic forgetting on old experiences.
        """
        if not old_experiences:
            return {'forgetting_score': 0.0, 'n_evaluated': 0}
        
        total_loss = 0.0
        n_experiences = 0
        
        for experience in old_experiences:
            inputs = experience['inputs']
            targets = experience['targets']
            
            outputs, _ = self.model.forward(inputs)
            loss = jnp.mean((outputs - targets) ** 2)
            total_loss += float(loss)
            n_experiences += 1
        
        avg_loss = total_loss / n_experiences if n_experiences > 0 else 0.0
        
        return {
            'forgetting_score': avg_loss,
            'n_evaluated': n_experiences
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        if not self.adaptation_history:
            return {}
        
        recent_metrics = self.adaptation_history[-10:]  # Last 10 steps
        
        stats = {
            'total_steps': len(self.adaptation_history),
            'memory_utilization': len(self.memory_buffer) / self.memory_size,
            'recent_avg_loss': np.mean([m['prediction_loss'] for m in recent_metrics]),
            'recent_avg_plasticity': np.mean([m['avg_plasticity'] for m in recent_metrics]),
            'importance_weight_stats': {}
        }
        
        # Importance weight statistics
        for key in self.importance_weights:
            weights = self.importance_weights[key]
            stats['importance_weight_stats'][key] = {
                'mean': float(jnp.mean(weights)),
                'std': float(jnp.std(weights)),
                'max': float(jnp.max(weights)),
                'min': float(jnp.min(weights))
            }
        
        return stats