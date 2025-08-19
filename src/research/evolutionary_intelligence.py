"""
Evolutionary Intelligence for Liquid Neural Networks.

This module implements advanced evolutionary algorithms for automatically
discovering optimal neural architectures, hyperparameters, and novel
algorithmic components for liquid neural networks.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit
    import equinox as eqx
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp

from typing import Dict, Any, Tuple, Optional, List, Callable, Union
import time
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import our models and components
from ..models.liquid_neural_network import LiquidNeuralNetwork
from ..models.continuous_time_rnn import ContinuousTimeRNN
from .novel_algorithms import MetaAdaptiveLiquidNetwork


@dataclass
class Individual:
    """Represents an individual in the evolutionary population."""
    genome: Dict[str, Any]
    fitness: Optional[float] = None
    age: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""
    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_ratio: float = 0.1
    tournament_size: int = 3
    max_network_size: int = 128
    min_network_size: int = 16
    fitness_aggregation: str = 'mean'  # 'mean', 'max', 'min', 'weighted'
    diversity_bonus: float = 0.05
    early_stopping_patience: int = 20


class ArchitectureSearchSpace:
    """Defines the search space for neural architecture evolution."""
    
    def __init__(self):
        self.architecture_types = [
            'liquid_neural_network',
            'continuous_time_rnn', 
            'meta_adaptive_liquid',
            'hybrid_architecture'
        ]
        
        self.activation_functions = ['tanh', 'relu', 'gelu', 'swish', 'sigmoid']
        self.optimizer_types = ['adam', 'sgd', 'rmsprop', 'adagrad']
        self.regularization_techniques = ['dropout', 'l1', 'l2', 'spectral_norm', 'none']
        
        # Architecture-specific parameters
        self.param_ranges = {
            'hidden_size': (16, 128),
            'num_layers': (1, 4),
            'sparsity_level': (0.05, 0.3),
            'tau_min': (0.1, 1.0),
            'tau_max': (2.0, 10.0),
            'learning_rate': (1e-5, 1e-2),
            'batch_size': (16, 128),
            'sequence_length': (50, 500),
            'sensory_mu': (0.1, 2.0),
            'sensory_sigma': (0.01, 0.5),
            'adaptation_rate': (0.001, 0.1),
            'plasticity_strength': (0.01, 0.5),
            'integration_dt': (0.01, 0.2)
        }
    
    def sample_random_genome(self, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Sample a random genome from the search space."""
        keys = random.split(key, 10)
        
        genome = {
            'architecture_type': random.choice(keys[0], jnp.array(self.architecture_types)),
            'activation': random.choice(keys[1], jnp.array(self.activation_functions)),
            'optimizer': random.choice(keys[2], jnp.array(self.optimizer_types)),
            'regularization': random.choice(keys[3], jnp.array(self.regularization_techniques))
        }
        
        # Sample continuous parameters
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if param_name in ['hidden_size', 'num_layers', 'batch_size', 'sequence_length']:
                # Integer parameters
                genome[param_name] = int(random.randint(keys[4], (), min_val, max_val + 1))
            else:
                # Continuous parameters (log-uniform for learning rate)
                if param_name == 'learning_rate':
                    log_lr = random.uniform(keys[5], (), jnp.log(min_val), jnp.log(max_val))
                    genome[param_name] = float(jnp.exp(log_lr))
                else:
                    genome[param_name] = float(random.uniform(keys[6], (), min_val, max_val))
        
        return genome
    
    def mutate_genome(self, genome: Dict[str, Any], mutation_rate: float, key: jax.random.PRNGKey) -> Tuple[Dict[str, Any], List[str]]:
        """Mutate a genome with given mutation rate."""
        keys = random.split(key, len(genome))
        new_genome = genome.copy()
        mutations_applied = []
        
        key_idx = 0
        for param_name, param_value in genome.items():
            if random.uniform(keys[key_idx]) < mutation_rate:
                if param_name == 'architecture_type':
                    new_genome[param_name] = random.choice(keys[key_idx], jnp.array(self.architecture_types))
                    mutations_applied.append(f"architecture_type -> {new_genome[param_name]}")
                
                elif param_name == 'activation':
                    new_genome[param_name] = random.choice(keys[key_idx], jnp.array(self.activation_functions))
                    mutations_applied.append(f"activation -> {new_genome[param_name]}")
                
                elif param_name == 'optimizer':
                    new_genome[param_name] = random.choice(keys[key_idx], jnp.array(self.optimizer_types))
                    mutations_applied.append(f"optimizer -> {new_genome[param_name]}")
                
                elif param_name == 'regularization':
                    new_genome[param_name] = random.choice(keys[key_idx], jnp.array(self.regularization_techniques))
                    mutations_applied.append(f"regularization -> {new_genome[param_name]}")
                
                elif param_name in self.param_ranges:
                    min_val, max_val = self.param_ranges[param_name]
                    
                    if param_name in ['hidden_size', 'num_layers', 'batch_size', 'sequence_length']:
                        # Integer mutation with Gaussian noise
                        noise = int(random.normal(keys[key_idx]) * 0.1 * (max_val - min_val))
                        new_val = int(jnp.clip(param_value + noise, min_val, max_val))
                        new_genome[param_name] = new_val
                        mutations_applied.append(f"{param_name}: {param_value} -> {new_val}")
                    
                    else:
                        # Continuous mutation with Gaussian noise
                        if param_name == 'learning_rate':
                            # Log-space mutation for learning rate
                            log_current = jnp.log(param_value)
                            log_noise = random.normal(keys[key_idx]) * 0.2
                            new_log = jnp.clip(log_current + log_noise, jnp.log(min_val), jnp.log(max_val))
                            new_val = float(jnp.exp(new_log))
                        else:
                            noise = random.normal(keys[key_idx]) * 0.1 * (max_val - min_val)
                            new_val = float(jnp.clip(param_value + noise, min_val, max_val))
                        
                        new_genome[param_name] = new_val
                        mutations_applied.append(f"{param_name}: {param_value:.4f} -> {new_val:.4f}")
            
            key_idx += 1
        
        return new_genome, mutations_applied
    
    def crossover_genomes(self, parent1: Dict[str, Any], parent2: Dict[str, Any], key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Create offspring by crossing over two parent genomes."""
        keys = random.split(key, len(parent1))
        offspring = {}
        
        key_idx = 0
        for param_name in parent1.keys():
            # Random selection from either parent
            if random.uniform(keys[key_idx]) < 0.5:
                offspring[param_name] = parent1[param_name]
            else:
                offspring[param_name] = parent2[param_name]
            
            # For continuous parameters, also try interpolation
            if param_name in self.param_ranges and param_name not in ['hidden_size', 'num_layers', 'batch_size', 'sequence_length']:
                if random.uniform(keys[key_idx]) < 0.3:  # 30% chance of interpolation
                    alpha = random.uniform(keys[key_idx], (), 0.2, 0.8)
                    val1, val2 = parent1[param_name], parent2[param_name]
                    
                    if param_name == 'learning_rate':
                        # Interpolate in log space
                        log_val = alpha * jnp.log(val1) + (1 - alpha) * jnp.log(val2)
                        offspring[param_name] = float(jnp.exp(log_val))
                    else:
                        offspring[param_name] = float(alpha * val1 + (1 - alpha) * val2)
            
            key_idx += 1
        
        return offspring


class FitnessEvaluator:
    """Evaluates the fitness of individuals in the population."""
    
    def __init__(
        self,
        benchmark_tasks: List[Callable],
        evaluation_budget: int = 100,
        fitness_weights: Optional[Dict[str, float]] = None
    ):
        self.benchmark_tasks = benchmark_tasks
        self.evaluation_budget = evaluation_budget
        self.fitness_weights = fitness_weights or {'performance': 0.7, 'efficiency': 0.2, 'complexity': 0.1}
        
    def evaluate_individual(self, individual: Individual, key: jax.random.PRNGKey) -> float:
        """Evaluate the fitness of an individual."""
        genome = individual.genome
        
        try:
            # Create model from genome
            model = self._create_model_from_genome(genome, key)
            
            # Evaluate on benchmark tasks
            performance_scores = []
            execution_times = []
            
            for task in self.benchmark_tasks:
                start_time = time.time()
                task_score = task(model, genome)
                execution_time = time.time() - start_time
                
                performance_scores.append(task_score)
                execution_times.append(execution_time)
            
            # Aggregate performance
            avg_performance = float(jnp.mean(jnp.array(performance_scores)))
            avg_execution_time = float(jnp.mean(jnp.array(execution_times)))
            
            # Compute efficiency score (inverse of execution time)
            efficiency_score = 1.0 / (1.0 + avg_execution_time)
            
            # Compute complexity penalty
            model_complexity = self._compute_model_complexity(genome)
            complexity_score = 1.0 / (1.0 + model_complexity)
            
            # Weighted fitness
            fitness = (
                self.fitness_weights['performance'] * avg_performance +
                self.fitness_weights['efficiency'] * efficiency_score +
                self.fitness_weights['complexity'] * complexity_score
            )
            
            # Add to performance history
            individual.performance_history.append(avg_performance)
            
            return float(fitness)
            
        except Exception as e:
            warnings.warn(f"Fitness evaluation failed: {str(e)}")
            return 0.0
    
    def _create_model_from_genome(self, genome: Dict[str, Any], key: jax.random.PRNGKey) -> Any:
        """Create a model from a genome specification."""
        arch_type = genome['architecture_type']
        hidden_size = genome['hidden_size']
        
        # Standard dimensions for evaluation
        input_size, output_size = 10, 1
        
        if arch_type == 'liquid_neural_network':
            return LiquidNeuralNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                sparsity_level=genome['sparsity_level'],
                tau_min=genome['tau_min'],
                tau_max=genome['tau_max'],
                sensory_mu=genome['sensory_mu'],
                sensory_sigma=genome['sensory_sigma'],
                key=key
            )
        
        elif arch_type == 'continuous_time_rnn':
            return ContinuousTimeRNN(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                dt=genome['integration_dt'],
                key=key
            )
        
        elif arch_type == 'meta_adaptive_liquid':
            return MetaAdaptiveLiquidNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                meta_learning_rate=genome['adaptation_rate'],
                plasticity_strength=genome['plasticity_strength'],
                key=key
            )
        
        else:
            # Default to liquid neural network
            return LiquidNeuralNetwork(input_size, hidden_size, output_size, key=key)
    
    def _compute_model_complexity(self, genome: Dict[str, Any]) -> float:
        """Compute model complexity score."""
        complexity = 0.0
        
        # Parameter count approximation
        hidden_size = genome['hidden_size']
        num_layers = genome.get('num_layers', 1)
        
        # Rough parameter count estimation
        param_count = hidden_size * (10 + hidden_size + 1) * num_layers
        complexity += param_count / 10000.0  # Normalize
        
        # Architecture complexity
        if genome['architecture_type'] == 'meta_adaptive_liquid':
            complexity += 1.5  # More complex architecture
        elif genome['architecture_type'] == 'continuous_time_rnn':
            complexity += 1.2
        
        return complexity


class EvolutionaryOptimizer:
    """Main evolutionary optimizer for neural architecture search."""
    
    def __init__(
        self,
        search_space: ArchitectureSearchSpace,
        fitness_evaluator: FitnessEvaluator,
        config: EvolutionConfig,
        random_seed: int = 42
    ):
        self.search_space = search_space
        self.fitness_evaluator = fitness_evaluator
        self.config = config
        self.random_seed = random_seed
        
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.diversity_history = []
        
    def initialize_population(self, key: jax.random.PRNGKey) -> None:
        """Initialize the evolutionary population."""
        keys = random.split(key, self.config.population_size)
        
        self.population = []
        for i in range(self.config.population_size):
            genome = self.search_space.sample_random_genome(keys[i])
            individual = Individual(genome=genome, age=0)
            self.population.append(individual)
    
    def evaluate_population(self, key: jax.random.PRNGKey) -> None:
        """Evaluate fitness for all individuals in the population."""
        keys = random.split(key, len(self.population))
        
        for i, individual in enumerate(self.population):
            if individual.fitness is None:  # Only evaluate if not already done
                individual.fitness = self.fitness_evaluator.evaluate_individual(individual, keys[i])
        
        # Update best individual
        best_idx = jnp.argmax(jnp.array([ind.fitness for ind in self.population]))
        if self.best_individual is None or self.population[best_idx].fitness > self.best_individual.fitness:
            self.best_individual = self.population[best_idx]
    
    def tournament_selection(self, key: jax.random.PRNGKey) -> Individual:
        """Select individual using tournament selection."""
        tournament_indices = random.choice(
            key, 
            len(self.population), 
            (self.config.tournament_size,), 
            replace=False
        )
        
        tournament_individuals = [self.population[i] for i in tournament_indices]
        return max(tournament_individuals, key=lambda ind: ind.fitness)
    
    def compute_diversity(self) -> float:
        """Compute population diversity based on genome differences."""
        if len(self.population) < 2:
            return 0.0
        
        diversity_scores = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                genome1, genome2 = self.population[i].genome, self.population[j].genome
                
                # Compare genomes
                differences = 0
                total_params = 0
                
                for key in genome1.keys():
                    total_params += 1
                    if isinstance(genome1[key], (int, float)):
                        if key in self.search_space.param_ranges:
                            min_val, max_val = self.search_space.param_ranges[key]
                            normalized_diff = abs(genome1[key] - genome2[key]) / (max_val - min_val)
                            differences += normalized_diff
                        else:
                            differences += abs(genome1[key] - genome2[key])
                    else:
                        differences += 1 if genome1[key] != genome2[key] else 0
                
                diversity_score = differences / max(total_params, 1)
                diversity_scores.append(diversity_score)
        
        return float(jnp.mean(jnp.array(diversity_scores)))
    
    def evolve_generation(self, key: jax.random.PRNGKey) -> None:
        """Evolve the population for one generation."""
        keys = random.split(key, 4)
        
        # Sort population by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Elite individuals (keep best unchanged)
        num_elites = int(self.config.elitism_ratio * self.config.population_size)
        new_population = self.population[:num_elites].copy()
        
        # Create offspring to fill rest of population
        offspring_keys = random.split(keys[0], self.config.population_size - num_elites)
        
        for i in range(self.config.population_size - num_elites):
            if random.uniform(keys[1]) < self.config.crossover_rate:
                # Crossover
                parent1 = self.tournament_selection(keys[2])
                parent2 = self.tournament_selection(keys[3])
                
                offspring_genome = self.search_space.crossover_genomes(
                    parent1.genome, parent2.genome, offspring_keys[i]
                )
                
                # Mutation
                offspring_genome, mutations = self.search_space.mutate_genome(
                    offspring_genome, self.config.mutation_rate, offspring_keys[i]
                )
                
                offspring = Individual(
                    genome=offspring_genome,
                    age=0,
                    parent_ids=[id(parent1), id(parent2)],
                    mutations=mutations
                )
                
            else:
                # Mutation only
                parent = self.tournament_selection(keys[2])
                offspring_genome, mutations = self.search_space.mutate_genome(
                    parent.genome.copy(), self.config.mutation_rate, offspring_keys[i]
                )
                
                offspring = Individual(
                    genome=offspring_genome,
                    age=0,
                    parent_ids=[id(parent)],
                    mutations=mutations
                )
            
            new_population.append(offspring)
        
        # Age existing individuals
        for individual in self.population:
            individual.age += 1
        
        self.population = new_population
        self.generation += 1
    
    def run_evolution(self) -> Dict[str, Any]:
        """Run the complete evolutionary optimization."""
        key = random.PRNGKey(self.random_seed)
        keys = random.split(key, self.config.num_generations + 1)
        
        # Initialize population
        self.initialize_population(keys[0])
        
        # Evolution loop
        best_fitness_stagnant = 0
        
        for gen in range(self.config.num_generations):
            print(f"Generation {gen + 1}/{self.config.num_generations}")
            
            # Evaluate population
            self.evaluate_population(keys[gen + 1])
            
            # Record statistics
            fitness_values = [ind.fitness for ind in self.population]
            diversity = self.compute_diversity()
            
            self.fitness_history.append({
                'generation': gen,
                'best_fitness': max(fitness_values),
                'mean_fitness': float(jnp.mean(jnp.array(fitness_values))),
                'std_fitness': float(jnp.std(jnp.array(fitness_values))),
                'worst_fitness': min(fitness_values)
            })
            
            self.diversity_history.append(diversity)
            
            print(f"  Best fitness: {max(fitness_values):.4f}")
            print(f"  Mean fitness: {jnp.mean(jnp.array(fitness_values)):.4f}")
            print(f"  Diversity: {diversity:.4f}")
            
            # Early stopping check
            if gen > 0 and max(fitness_values) <= self.fitness_history[-2]['best_fitness']:
                best_fitness_stagnant += 1
            else:
                best_fitness_stagnant = 0
            
            if best_fitness_stagnant >= self.config.early_stopping_patience:
                print(f"Early stopping at generation {gen + 1}")
                break
            
            # Evolve to next generation
            if gen < self.config.num_generations - 1:
                self.evolve_generation(keys[gen + 1])
        
        # Final evaluation
        self.evaluate_population(keys[-1])
        
        return {
            'best_individual': self.best_individual,
            'final_population': self.population,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'total_generations': self.generation,
            'search_space_coverage': self.compute_search_space_coverage()
        }
    
    def compute_search_space_coverage(self) -> Dict[str, Any]:
        """Compute how well the search space was explored."""
        coverage_stats = {}
        
        # Analyze parameter distributions in final population
        for param_name in self.search_space.param_ranges.keys():
            values = [ind.genome[param_name] for ind in self.population]
            min_val, max_val = self.search_space.param_ranges[param_name]
            
            coverage_stats[param_name] = {
                'explored_range': (min(values), max(values)),
                'full_range': (min_val, max_val),
                'coverage_ratio': (max(values) - min(values)) / (max_val - min_val),
                'mean': float(jnp.mean(jnp.array(values))),
                'std': float(jnp.std(jnp.array(values)))
            }
        
        # Analyze categorical parameter distributions
        categorical_params = ['architecture_type', 'activation', 'optimizer', 'regularization']
        for param_name in categorical_params:
            values = [ind.genome[param_name] for ind in self.population]
            unique_values = list(set(values))
            
            coverage_stats[param_name] = {
                'unique_values_explored': unique_values,
                'coverage_ratio': len(unique_values) / len(getattr(self.search_space, param_name + 's', [])),
                'distribution': {val: values.count(val) for val in unique_values}
            }
        
        return coverage_stats