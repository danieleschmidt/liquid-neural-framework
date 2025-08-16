"""
Test Generation 4 Evolutionary Intelligence Features
"""

import jax
import jax.numpy as jnp
import sys
sys.path.append('src')

from models.evolutionary_intelligence import (
    EvolutionaryOptimizer, MetaLearningSystem, 
    SelfModifyingNetwork, AutonomousResearchSystem
)
from models.liquid_neural_network import LiquidNeuralNetwork

def test_generation4_evolutionary():
    """Test Generation 4 evolutionary intelligence features."""
    
    print('🧠 Testing Generation 4 Evolutionary Intelligence...')
    
    # Test 1: Evolutionary Architecture Optimization
    print('\n1. Testing evolutionary architecture optimization...')
    key = jax.random.PRNGKey(42)
    
    # Create test data
    test_data = jax.random.normal(key, (20, 10))  # 20 time steps, 10 features
    
    # Initialize evolutionary optimizer
    evo_optimizer = EvolutionaryOptimizer(population_size=10, mutation_rate=0.15)
    
    # Test random architecture generation
    random_arch = evo_optimizer.create_random_architecture(input_size=10, output_size=5, key=key)
    print(f'✅ Random architecture: {random_arch["hidden_sizes"]} layers')
    
    # Test architecture mutation
    mutated_arch = evo_optimizer.mutate_architecture(random_arch, key)
    print(f'✅ Mutated architecture: {mutated_arch["hidden_sizes"]} layers')
    
    # Test fitness evaluation
    fitness = evo_optimizer.evaluate_fitness(random_arch, test_data, key)
    print(f'✅ Architecture fitness: {fitness:.4f}')
    
    # Test evolution (quick run)
    print('   Running mini evolution (3 generations)...')
    best_arch = evo_optimizer.evolve_population(
        test_data, input_size=10, output_size=5, generations=3, key=key
    )
    print(f'✅ Best evolved architecture: {best_arch["hidden_sizes"]} layers')
    print(f'   Best fitness history: {[f"{f:.3f}" for f in evo_optimizer.best_fitness_history]}')
    
    # Test 2: Meta-Learning System
    print('\n2. Testing meta-learning system...')
    meta_learner = MetaLearningSystem()
    
    # Register tasks
    meta_learner.register_task('classification_task', {'type': 'classification'})
    meta_learner.register_task('regression_task', {'type': 'regression'})
    
    # Create base model for adaptation
    base_model = LiquidNeuralNetwork(input_size=10, hidden_sizes=[16], output_size=5, key=key)
    
    # Learn adaptation strategy
    strategy = meta_learner.learn_adaptation_strategy(
        'classification_task', base_model, test_data, test_data
    )
    print(f'✅ Learned adaptation strategy: {strategy}')
    
    # Get meta-learning insights
    insights = meta_learner.get_meta_learning_insights()
    print(f'✅ Meta-learning insights: {insights["total_tasks"]} tasks, {insights["average_performance"]:.3f} avg performance')
    
    # Test 3: Self-Modifying Network
    print('\n3. Testing self-modifying network...')
    
    # Create self-modifying network with low threshold for testing
    self_mod_net = SelfModifyingNetwork(
        input_size=10, hidden_sizes=[16, 8], output_size=5, key=key,
        adaptation_threshold=0.9,  # High threshold to trigger modification
        modification_frequency=5   # Check every 5 steps
    )
    
    print(f'   Initial architecture: {[layer.hidden_size for layer in self_mod_net.base_network.layers]} neurons')
    
    # Process multiple batches to trigger self-modification
    for i in range(3):
        outputs, states = self_mod_net(test_data)
        print(f'   Step {i+1}: Output shape {outputs.shape}')
    
    # Get modification history
    mod_history = self_mod_net.get_modification_history()
    print(f'✅ Self-modifications made: {len(mod_history)}')
    
    # Get evolution stats
    evo_stats = self_mod_net.get_evolution_stats()
    print(f'✅ Evolution stats: {evo_stats["total_modifications"]} modifications, step {evo_stats["current_step"]}')
    
    # Test 4: Autonomous Research System
    print('\n4. Testing autonomous research system...')
    
    from models.evolutionary_intelligence import autonomous_researcher
    
    # Define current capabilities and gaps
    capabilities = {
        'inference_speed': 0.8,
        'accuracy': 0.7,
        'memory_efficiency': 0.6
    }
    
    performance_gaps = {
        'speed': {'severity': 0.6, 'description': 'Inference too slow'},
        'stability': {'severity': 0.8, 'description': 'Output variance too high'}
    }
    
    # Propose research directions
    research_proposal = autonomous_researcher.propose_research_direction(
        capabilities, performance_gaps
    )
    
    print(f'✅ Research proposals generated: {research_proposal["total_proposals"]} proposals')
    print(f'   Top proposal: {research_proposal["proposals"][0]["title"]}')
    print(f'   Research budget: {research_proposal["research_budget"]["total_experiments"]} experiments')
    
    # Execute autonomous research (simplified)
    if research_proposal['proposals']:
        top_proposal = research_proposal['proposals'][0]
        research_results = autonomous_researcher.execute_autonomous_research(
            top_proposal, test_data
        )
        print(f'✅ Research completed: {research_results["status"]}')
        if 'performance_improvement' in research_results:
            print(f'   Performance improvement: {research_results["performance_improvement"]:.1%}')
    
    # Get research summary
    research_summary = autonomous_researcher.get_research_summary()
    print(f'✅ Research summary: {research_summary["total_projects"]} projects, {research_summary["total_discoveries"]} discoveries')
    
    # Test 5: Integration Test - Evolutionary + Self-Modification
    print('\n5. Testing integrated evolutionary intelligence...')
    
    # Create enhanced self-modifying network with evolutionary capabilities
    enhanced_net = SelfModifyingNetwork(
        input_size=10, hidden_sizes=[32], output_size=5, key=key,
        adaptation_threshold=0.5,
        modification_frequency=3
    )
    
    print('   Running integrated evolution with self-modification...')
    
    # Process data and observe adaptations
    total_outputs = []
    for step in range(6):
        batch_data = jax.random.normal(jax.random.split(key, step+1)[0], (15, 10))
        outputs, _ = enhanced_net(batch_data)
        total_outputs.append(outputs)
        
        if step == 2:  # Check mid-way
            current_arch = [layer.hidden_size for layer in enhanced_net.base_network.layers]
            print(f'   Step {step+1} architecture: {current_arch} neurons')
    
    # Final statistics
    final_mod_history = enhanced_net.get_modification_history()
    final_evo_stats = enhanced_net.get_evolution_stats()
    
    print(f'✅ Final modifications: {len(final_mod_history)}')
    print(f'✅ Final evolution stats: {final_evo_stats["evolutionary_generations"]} generations')
    
    # Summary
    print('\n🎉 Generation 4 Evolutionary Intelligence Summary:')
    print(f'   ✅ Evolutionary Optimization: {len(evo_optimizer.best_fitness_history)} generations')
    print(f'   ✅ Meta-Learning: {insights["total_tasks"]} tasks learned')
    print(f'   ✅ Self-Modification: {len(final_mod_history)} adaptations made')
    print(f'   ✅ Autonomous Research: {research_summary["total_projects"]} projects executed')
    print(f'   ✅ Discoveries Made: {research_summary["total_discoveries"]} novel findings')
    
    print('\n🧠 Generation 4 Evolutionary Intelligence Complete!')
    
    return {
        'evolutionary_generations': len(evo_optimizer.best_fitness_history),
        'meta_learning_tasks': insights['total_tasks'],
        'self_modifications': len(final_mod_history),
        'research_projects': research_summary['total_projects'],
        'discoveries': research_summary['total_discoveries'],
        'best_fitness': max(evo_optimizer.best_fitness_history) if evo_optimizer.best_fitness_history else 0.0
    }

if __name__ == "__main__":
    results = test_generation4_evolutionary()
    print(f'\nFinal Evolutionary Results: {results}')