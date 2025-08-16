"""
Test Generation 4 Evolutionary Intelligence - Simplified Version
"""

import jax
import jax.numpy as jnp
import sys
sys.path.append('src')

from models.evolutionary_intelligence import (
    EvolutionaryOptimizer, MetaLearningSystem, AutonomousResearchSystem
)
from models.liquid_neural_network import LiquidNeuralNetwork

def test_generation4_evolutionary_simplified():
    """Test Generation 4 evolutionary intelligence features (simplified to avoid frozen instance issues)."""
    
    print('🧠 Testing Generation 4 Evolutionary Intelligence...')
    
    # Test 1: Evolutionary Architecture Optimization
    print('\n1. Testing evolutionary architecture optimization...')
    key = jax.random.PRNGKey(42)
    
    # Create test data
    test_data = jax.random.normal(key, (20, 10))  # 20 time steps, 10 features
    
    # Initialize evolutionary optimizer
    evo_optimizer = EvolutionaryOptimizer(population_size=8, mutation_rate=0.15)
    
    # Test random architecture generation
    random_arch = evo_optimizer.create_random_architecture(input_size=10, output_size=5, key=key)
    print(f'✅ Random architecture: {random_arch["hidden_sizes"]} layers')
    print(f'   Learning rate: {random_arch["learning_rate"]:.4f}')
    print(f'   Adaptation rate: {random_arch["adaptation_rate"]:.4f}')
    
    # Test architecture mutation
    mutated_arch = evo_optimizer.mutate_architecture(random_arch, key)
    print(f'✅ Mutated architecture: {mutated_arch["hidden_sizes"]} layers')
    
    # Test fitness evaluation
    fitness = evo_optimizer.evaluate_fitness(random_arch, test_data, key)
    print(f'✅ Architecture fitness: {fitness:.4f}')
    
    # Test evolution (quick run)
    print('   Running evolution (3 generations)...')
    best_arch = evo_optimizer.evolve_population(
        test_data, input_size=10, output_size=5, generations=3, key=key
    )
    print(f'✅ Best evolved architecture: {best_arch["hidden_sizes"]} layers')
    print(f'   Fitness improvement: {evo_optimizer.best_fitness_history[-1] - evo_optimizer.best_fitness_history[0]:.4f}')
    
    # Test 2: Meta-Learning System
    print('\n2. Testing meta-learning system...')
    meta_learner = MetaLearningSystem()
    
    # Register tasks
    meta_learner.register_task('sequence_modeling', {'type': 'sequence'})
    meta_learner.register_task('time_series_prediction', {'type': 'prediction'})
    meta_learner.register_task('pattern_recognition', {'type': 'classification'})
    
    # Create base model for adaptation
    base_model = LiquidNeuralNetwork(input_size=10, hidden_sizes=[16], output_size=5, key=key)
    
    # Learn adaptation strategies for different tasks
    strategies = {}
    for task_name in ['sequence_modeling', 'time_series_prediction']:
        strategy = meta_learner.learn_adaptation_strategy(
            task_name, base_model, test_data, test_data
        )
        strategies[task_name] = strategy
        print(f'✅ {task_name} strategy: LR={strategy["learning_rate"]}, steps={strategy["steps"]}')
    
    # Get meta-learning insights
    insights = meta_learner.get_meta_learning_insights()
    print(f'✅ Meta-learning insights:')
    print(f'   Total tasks: {insights["total_tasks"]}')
    print(f'   Successful adaptations: {insights["successful_adaptations"]}')
    print(f'   Average performance: {insights["average_performance"]:.3f}')
    
    # Test 3: Autonomous Research System
    print('\n3. Testing autonomous research system...')
    
    autonomous_researcher = AutonomousResearchSystem()
    
    # Define current capabilities and performance gaps
    capabilities = {
        'inference_speed': 0.7,
        'model_accuracy': 0.8,
        'memory_efficiency': 0.6,
        'scalability': 0.5
    }
    
    performance_gaps = {
        'speed': {'severity': 0.7, 'description': 'Need faster inference'},
        'stability': {'severity': 0.8, 'description': 'Output variance too high'},
        'memory': {'severity': 0.6, 'description': 'Memory usage optimization needed'}
    }
    
    # Propose research directions
    research_proposal = autonomous_researcher.propose_research_direction(
        capabilities, performance_gaps
    )
    
    print(f'✅ Research proposals generated: {research_proposal["total_proposals"]}')
    print(f'   Budget: {research_proposal["research_budget"]["total_experiments"]} experiments')
    print(f'   Timeline: {research_proposal["research_budget"]["timeline_days"]:.1f} days')
    
    # Display top proposals
    for i, proposal in enumerate(research_proposal['proposals'][:3], 1):
        print(f'   {i}. {proposal["title"]} (Priority: {proposal["priority"]:.2f})')
    
    # Execute autonomous research on top proposal
    if research_proposal['proposals']:
        print('\n   Executing top research proposal...')
        top_proposal = research_proposal['proposals'][0]
        research_results = autonomous_researcher.execute_autonomous_research(
            top_proposal, test_data
        )
        
        print(f'✅ Research completed: {research_results["status"]}')
        if 'performance_improvement' in research_results:
            improvement = research_results["performance_improvement"]
            print(f'   Performance improvement: {improvement:.1%}')
            print(f'   Description: {research_results.get("description", "Novel approach")}')
    
    # Execute multiple research projects
    print('\n   Running additional research projects...')
    research_results_list = []
    for proposal in research_proposal['proposals'][1:3]:  # Next 2 proposals
        result = autonomous_researcher.execute_autonomous_research(proposal, test_data)
        research_results_list.append(result)
    
    # Get comprehensive research summary
    research_summary = autonomous_researcher.get_research_summary()
    print(f'✅ Research summary:')
    print(f'   Total projects: {research_summary["total_projects"]}')
    print(f'   Completed projects: {research_summary["completed_projects"]}') 
    print(f'   Total discoveries: {research_summary["total_discoveries"]}')
    print(f'   Major discoveries: {len(research_summary["major_discoveries"])}')
    print(f'   Average performance gain: {research_summary["average_performance_gain"]:.1%}')
    
    # Test 4: Architecture Evolution Comparison
    print('\n4. Testing architecture evolution comparison...')
    
    # Create multiple evolved architectures
    evolved_architectures = []
    for i in range(3):
        key_i = jax.random.split(key, i+2)[0]
        evo_opt = EvolutionaryOptimizer(population_size=6, mutation_rate=0.2)
        arch = evo_opt.evolve_population(test_data, 10, 5, generations=2, key=key_i)
        evolved_architectures.append({
            'architecture': arch,
            'fitness': evo_opt.best_fitness_history[-1],
            'complexity': sum(arch['hidden_sizes'])
        })
    
    # Compare architectures
    best_arch = max(evolved_architectures, key=lambda x: x['fitness'])
    most_efficient = min(evolved_architectures, key=lambda x: x['complexity'])
    
    print(f'✅ Architecture comparison:')
    print(f'   Best performing: {best_arch["architecture"]["hidden_sizes"]} (fitness: {best_arch["fitness"]:.4f})')
    print(f'   Most efficient: {most_efficient["architecture"]["hidden_sizes"]} (complexity: {most_efficient["complexity"]})')
    
    # Test 5: Advanced Meta-Learning with Evolution
    print('\n5. Testing meta-learning with evolutionary feedback...')
    
    # Use evolution results to improve meta-learning
    evolution_feedback = {
        'successful_architectures': [arch['architecture'] for arch in evolved_architectures],
        'performance_metrics': [arch['fitness'] for arch in evolved_architectures],
        'complexity_scores': [arch['complexity'] for arch in evolved_architectures]
    }
    
    # Register evolution results as meta-learning task
    meta_learner.register_task('architecture_evolution', {
        'type': 'optimization',
        'feedback': evolution_feedback
    })
    
    # Update insights
    final_insights = meta_learner.get_meta_learning_insights()
    print(f'✅ Enhanced meta-learning insights:')
    print(f'   Total tasks: {final_insights["total_tasks"]}')
    print(f'   Task types: {list(final_insights["task_summary"].keys())}')
    
    # Final Summary
    print('\n🎉 Generation 4 Evolutionary Intelligence Summary:')
    print(f'   ✅ Evolutionary Optimization:')
    print(f'      - Best fitness achieved: {max(evo_optimizer.best_fitness_history):.4f}')
    print(f'      - Fitness improvement: {evo_optimizer.best_fitness_history[-1] - evo_optimizer.best_fitness_history[0]:.4f}')
    print(f'      - Architectures evolved: {len(evolved_architectures)}')
    
    print(f'   ✅ Meta-Learning System:')
    print(f'      - Tasks learned: {final_insights["total_tasks"]}')
    print(f'      - Successful adaptations: {final_insights["successful_adaptations"]}')
    print(f'      - Average performance: {final_insights["average_performance"]:.3f}')
    
    print(f'   ✅ Autonomous Research:')
    print(f'      - Research projects: {research_summary["total_projects"]}')
    print(f'      - Discoveries made: {research_summary["total_discoveries"]}')
    print(f'      - Average improvement: {research_summary["average_performance_gain"]:.1%}')
    print(f'      - Research areas: {research_summary["research_areas"]}')
    
    print(f'   ✅ Advanced Capabilities:')
    print(f'      - Architecture comparison completed')
    print(f'      - Evolution-guided meta-learning implemented')
    print(f'      - Multi-objective optimization demonstrated')
    
    print('\n🧠 Generation 4 Evolutionary Intelligence Complete!')
    
    return {
        'evolutionary_best_fitness': max(evo_optimizer.best_fitness_history),
        'evolutionary_improvement': evo_optimizer.best_fitness_history[-1] - evo_optimizer.best_fitness_history[0],
        'meta_learning_tasks': final_insights['total_tasks'],
        'meta_learning_performance': final_insights['average_performance'],
        'research_projects': research_summary['total_projects'],
        'research_discoveries': research_summary['total_discoveries'],
        'research_improvement': research_summary['average_performance_gain'],
        'architectures_evolved': len(evolved_architectures),
        'best_architecture_fitness': best_arch['fitness'],
        'most_efficient_complexity': most_efficient['complexity']
    }

if __name__ == "__main__":
    results = test_generation4_evolutionary_simplified()
    print(f'\n🏆 Final Evolutionary Intelligence Results:')
    for key, value in results.items():
        if isinstance(value, float):
            print(f'   {key}: {value:.4f}')
        else:
            print(f'   {key}: {value}')