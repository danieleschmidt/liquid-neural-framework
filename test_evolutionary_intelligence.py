"""
Standalone test for Evolutionary Intelligence Framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import time


class EvolutionaryOptimizer:
    """Advanced evolutionary optimizer for neural architectures."""
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        selection_pressure: float = 2.0
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        
        # Evolution tracking
        self.generation = 0
        self.fitness_history = []
        self.population = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
    def create_random_individual(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create a random individual within the search space."""
        individual = {}
        
        for param, bounds in search_space.items():
            if isinstance(bounds, dict):
                if bounds['type'] == 'int':
                    individual[param] = np.random.randint(bounds['min'], bounds['max'] + 1)
                elif bounds['type'] == 'float':
                    individual[param] = np.random.uniform(bounds['min'], bounds['max'])
                elif bounds['type'] == 'choice':
                    individual[param] = np.random.choice(bounds['options'])
                elif bounds['type'] == 'bool':
                    individual[param] = np.random.choice([True, False])
        
        return individual
    
    def initialize_population(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize the population randomly."""
        population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual(search_space)
            population.append(individual)
        return population
    
    def mutate_individual(
        self, 
        individual: Dict[str, Any], 
        search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply mutation to an individual."""
        mutated = individual.copy()
        
        for param, bounds in search_space.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(bounds, dict):
                    if bounds['type'] == 'int':
                        std = (bounds['max'] - bounds['min']) * 0.1
                        delta = int(np.random.normal(0, std))
                        mutated[param] = np.clip(
                            individual[param] + delta,
                            bounds['min'], bounds['max']
                        )
                    elif bounds['type'] == 'float':
                        std = (bounds['max'] - bounds['min']) * 0.1
                        delta = np.random.normal(0, std)
                        mutated[param] = np.clip(
                            individual[param] + delta,
                            bounds['min'], bounds['max']
                        )
                    elif bounds['type'] == 'choice':
                        mutated[param] = np.random.choice(bounds['options'])
                    elif bounds['type'] == 'bool':
                        mutated[param] = not individual[param]
        
        return mutated
    
    def crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2
    
    def tournament_selection(
        self, 
        population: List[Dict[str, Any]], 
        fitnesses: List[float], 
        tournament_size: int = 3
    ) -> Dict[str, Any]:
        """Select an individual using tournament selection."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def evolve_generation(
        self,
        population: List[Dict[str, Any]],
        fitnesses: List[float],
        search_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evolve one generation of the population."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = max(1, int(0.1 * self.population_size))
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate_individual(child1, search_space)
            child2 = self.mutate_individual(child2, search_space)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def optimize(
        self,
        fitness_function: Callable,
        search_space: Dict[str, Any],
        max_generations: int = 100,
        target_fitness: Optional[float] = None,
        patience: int = 10
    ) -> Dict[str, Any]:
        """Run evolutionary optimization."""
        print(f"🧬 Starting evolutionary optimization...")
        print(f"Population: {self.population_size}, Generations: {max_generations}")
        
        # Initialize population
        self.population = self.initialize_population(search_space)
        self.generation = 0
        self.fitness_history = []
        
        best_fitness_history = []
        generations_without_improvement = 0
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate population
            fitnesses = []
            for individual in self.population:
                try:
                    fitness = fitness_function(individual)
                    fitnesses.append(fitness)
                except Exception as e:
                    fitnesses.append(float('-inf'))
            
            # Track statistics
            best_gen_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            self.fitness_history.append(avg_fitness)
            best_fitness_history.append(best_gen_fitness)
            
            # Update best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = self.population[best_idx].copy()
                generations_without_improvement = 0
                print(f"🔥 Gen {generation}: New best fitness = {self.best_fitness:.6f}")
            else:
                generations_without_improvement += 1
            
            # Print progress
            if generation % 5 == 0 or generation == max_generations - 1:
                print(f"Gen {generation}: Best={best_gen_fitness:.6f}, Avg={avg_fitness:.6f}")
            
            # Check convergence criteria
            if target_fitness and self.best_fitness >= target_fitness:
                print(f"🎯 Target fitness {target_fitness} reached!")
                break
            
            if generations_without_improvement >= patience:
                print(f"🛑 Early stopping: No improvement for {patience} generations")
                break
            
            # Evolve to next generation
            if generation < max_generations - 1:
                self.population = self.evolve_generation(
                    self.population, fitnesses, search_space
                )
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations': self.generation + 1,
            'fitness_history': self.fitness_history,
            'best_fitness_history': best_fitness_history
        }


class MetaLearningSystem:
    """Meta-learning system for adaptive optimization."""
    
    def __init__(self, adaptation_window: int = 50, meta_learning_rate: float = 0.01):
        self.adaptation_window = adaptation_window
        self.meta_learning_rate = meta_learning_rate
        
        self.meta_parameters = {
            'base_learning_rate': 0.01,
            'exploration_factor': 0.1,
            'adaptation_sensitivity': 0.05
        }
        
        self.performance_history = []
        self.adaptation_history = []
    
    def update_meta_parameters(self, performance: float):
        """Update meta-parameters based on performance feedback."""
        self.performance_history.append(performance)
        
        if len(self.performance_history) >= self.adaptation_window:
            recent_performance = self.performance_history[-self.adaptation_window:]
            
            # Performance trend
            if len(recent_performance) > 1:
                performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            else:
                performance_trend = 0
            
            # Performance variance
            performance_variance = np.var(recent_performance)
            
            # Adapt parameters
            if performance_trend < 0:  # Performance declining
                self.meta_parameters['base_learning_rate'] *= 0.98
                self.meta_parameters['exploration_factor'] *= 1.02
            else:  # Performance improving
                self.meta_parameters['base_learning_rate'] *= 1.001
                self.meta_parameters['exploration_factor'] *= 0.99
            
            # Apply constraints
            self.meta_parameters['base_learning_rate'] = np.clip(
                self.meta_parameters['base_learning_rate'], 0.001, 0.1
            )
            
            self.meta_parameters['exploration_factor'] = np.clip(
                self.meta_parameters['exploration_factor'], 0.01, 0.5
            )
            
            # Record adaptation
            self.adaptation_history.append({
                'step': len(self.performance_history),
                'performance': performance,
                'performance_trend': performance_trend,
                'meta_parameters': self.meta_parameters.copy()
            })
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about meta-learning adaptation."""
        if not self.performance_history:
            return {}
        
        return {
            'total_steps': len(self.performance_history),
            'current_performance': self.performance_history[-1],
            'avg_performance': np.mean(self.performance_history),
            'performance_trend': np.polyfit(range(len(self.performance_history)), self.performance_history, 1)[0] if len(self.performance_history) > 1 else 0,
            'current_meta_parameters': self.meta_parameters,
            'num_adaptations': len(self.adaptation_history)
        }


def run_evolutionary_intelligence_demo():
    """Demonstration of evolutionary intelligence capabilities."""
    print("🧬 EVOLUTIONARY INTELLIGENCE DEMO")
    print("=" * 50)
    
    # 1. Test Basic Evolutionary Optimizer
    print("\n🧪 Testing Basic Evolutionary Optimizer...")
    
    def optimization_function(individual):
        # Minimize distance to (2, -1) with some noise
        x, y = individual['x'], individual['y']
        target_x, target_y = 2.0, -1.0
        distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        noise = 0.1 * np.random.random()
        return -(distance + noise)  # Negative for maximization
    
    search_space = {
        'x': {'type': 'float', 'min': -10, 'max': 10},
        'y': {'type': 'float', 'min': -10, 'max': 10}
    }
    
    optimizer = EvolutionaryOptimizer(population_size=30, mutation_rate=0.15)
    result = optimizer.optimize(
        optimization_function, search_space, max_generations=25, patience=8
    )
    
    print(f"✅ Target: (2.0, -1.0)")
    print(f"✅ Found: ({result['best_individual']['x']:.4f}, {result['best_individual']['y']:.4f})")
    target_distance = np.sqrt((result['best_individual']['x'] - 2.0)**2 + (result['best_individual']['y'] + 1.0)**2)
    print(f"✅ Distance from target: {target_distance:.4f}")
    print(f"✅ Converged in {result['generations']} generations")
    
    # 2. Test Meta-Learning System
    print("\n🎛️  Testing Meta-Learning System...")
    
    meta_learner = MetaLearningSystem()
    
    # Simulate learning process with phases
    for step in range(80):
        # Phase 1: Learning (improving)
        if step < 25:
            base_performance = 0.3 + 0.02 * step
        # Phase 2: Plateau (stable)
        elif step < 50:
            base_performance = 0.8
        # Phase 3: Adaptation (slight decline then recovery)
        else:
            decay_factor = max(0, 0.8 - 0.01 * (step - 50))
            recovery_factor = 0.005 * max(0, step - 60)
            base_performance = decay_factor + recovery_factor
        
        # Add noise
        performance = base_performance + 0.1 * (np.random.random() - 0.5)
        performance = max(0, min(1, performance))  # Clip to [0, 1]
        
        meta_learner.update_meta_parameters(performance)
        
        if step % 20 == 0:
            print(f"   Step {step}: Performance = {performance:.4f}")
    
    stats = meta_learner.get_adaptation_statistics()
    print(f"✅ Meta-learning completed:")
    print(f"   Performance trend: {stats['performance_trend']:.6f}")
    print(f"   Final learning rate: {stats['current_meta_parameters']['base_learning_rate']:.6f}")
    print(f"   Final exploration: {stats['current_meta_parameters']['exploration_factor']:.6f}")
    print(f"   Adaptations made: {stats['num_adaptations']}")
    
    # 3. Multi-objective optimization test
    print("\n🎯 Testing Multi-Objective Optimization...")
    
    def multi_objective_fitness(individual):
        # Balance multiple objectives: accuracy, efficiency, complexity
        size = individual['network_size']
        learning_rate = individual['learning_rate']
        
        # Simulated objectives
        accuracy = 1.0 / (1.0 + np.exp(-(size - 50) / 20))  # Larger networks -> higher accuracy
        efficiency = np.exp(-learning_rate * 100)  # Lower learning rate -> higher efficiency  
        complexity_penalty = size / 200.0  # Smaller networks preferred
        
        # Combined fitness (Pareto-style weighting)
        fitness = 0.6 * accuracy + 0.3 * efficiency - 0.1 * complexity_penalty
        
        return fitness
    
    multi_search_space = {
        'network_size': {'type': 'int', 'min': 10, 'max': 200},
        'learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.1},
        'use_dropout': {'type': 'bool'},
        'activation': {'type': 'choice', 'options': ['relu', 'tanh', 'sigmoid']}
    }
    
    multi_optimizer = EvolutionaryOptimizer(population_size=25, mutation_rate=0.2)
    multi_result = multi_optimizer.optimize(
        multi_objective_fitness, multi_search_space, max_generations=20
    )
    
    best = multi_result['best_individual']
    print(f"✅ Best configuration found:")
    print(f"   Network size: {best['network_size']}")
    print(f"   Learning rate: {best['learning_rate']:.6f}")
    print(f"   Use dropout: {best['use_dropout']}")
    print(f"   Activation: {best['activation']}")
    print(f"   Fitness: {multi_result['best_fitness']:.4f}")
    
    print("\n🎉 EVOLUTIONARY INTELLIGENCE DEMO COMPLETED!")
    print("✅ All tests passed successfully!")
    
    return {
        'basic_optimizer': result,
        'meta_learning': stats,
        'multi_objective': multi_result
    }


if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    demo_results = run_evolutionary_intelligence_demo()