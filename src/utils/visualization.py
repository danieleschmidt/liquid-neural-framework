"""
Advanced visualization utilities for liquid neural networks.

Provides comprehensive visualization capabilities including:
- Network architecture diagrams
- Training dynamics plots
- Performance benchmarks
- Research-ready publication plots
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    patches = None
    FuncAnimation = None
    sns = None
    warnings.warn("Matplotlib not available. Visualization functionality will be limited.")

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # Fallback to numpy


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class ResultsVisualizer:
    """
    Publication-ready visualization of experimental results.
    
    Creates comprehensive plots for research papers and presentations
    with professional styling and statistical annotations.
    """
    
    def __init__(
        self,
        style: str = 'publication',
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        if not HAS_MATPLOTLIB:
            raise VisualizationError(
                "Matplotlib is required for visualization. Please install with: pip install matplotlib seaborn"
            )
        
        self._setup_style()
    
    def _setup_style(self):
        """Setup publication-ready plotting style."""
        if self.style == 'publication':
            plt.style.use('default')
            
            # Publication settings
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'axes.linewidth': 1.2,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'xtick.major.size': 6,
                'ytick.major.size': 6,
                'xtick.minor.size': 3,
                'ytick.minor.size': 3,
                'legend.frameon': False,
                'legend.fontsize': 10,
                'figure.dpi': self.dpi,
                'savefig.dpi': self.dpi,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
            
            # Set color palette
            self.colors = sns.color_palette("husl", 8)
            
        elif self.style == 'dark':
            plt.style.use('dark_background')
            self.colors = sns.color_palette("bright", 8)
            
        elif self.style == 'minimal':
            sns.set_style("whitegrid")
            self.colors = sns.color_palette("deep", 8)
    
    def plot_training_curves(
        self,
        histories: Dict[str, Dict[str, List[float]]],
        metrics: List[str] = ['train_loss', 'val_loss'],
        title: str = "Training Curves",
        save_path: Optional[str] = None,
        show_confidence: bool = True,
        log_scale: bool = False
    ) -> Optional[Any]:
        """
        Plot training curves with statistical analysis.
        
        Args:
            histories: Dictionary mapping model names to training histories
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the figure
            show_confidence: Whether to show confidence intervals
            log_scale: Whether to use log scale for y-axis
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for j, (model_name, history) in enumerate(histories.items()):
                if metric in history:
                    values = np.array(history[metric])
                    epochs = np.arange(1, len(values) + 1)
                    
                    # Plot main line
                    color = self.colors[j % len(self.colors)]
                    ax.plot(epochs, values, label=model_name, 
                           color=color, linewidth=2, alpha=0.8)
                    
                    # Add confidence interval if multiple runs
                    if show_confidence and len(values) > 10:
                        # Simple moving average confidence
                        window = max(1, len(values) // 10)
                        smoothed = np.convolve(values, np.ones(window)/window, mode='same')
                        std = np.std(values - smoothed)
                        
                        ax.fill_between(epochs, 
                                      smoothed - std, 
                                      smoothed + std, 
                                      alpha=0.2, color=color)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            
            if log_scale:
                ax.set_yscale('log')
            
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Performance Comparison",
        save_path: Optional[str] = None,
        include_error_bars: bool = True,
        show_values: bool = True
    ) -> Optional[Any]:
        """
        Create comprehensive performance comparison plots.
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(results.keys())
        x_pos = np.arange(len(model_names))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract values and errors
            values = []
            errors = []
            
            for model_name in model_names:
                if metric in results[model_name]:
                    val = results[model_name][metric]
                    if isinstance(val, dict) and 'mean' in val:
                        values.append(val['mean'])
                        errors.append(val.get('std', 0))
                    else:
                        values.append(float(val))
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Create bars
            bars = ax.bar(x_pos, values, 
                         yerr=errors if include_error_bars else None,
                         capsize=5,
                         color=self.colors[:len(model_names)],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1)
            
            # Add value labels on bars
            if show_values:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.1,
                           f'{value:.3f}',
                           ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight best performance
            if values:
                best_idx = np.argmin(values) if 'loss' in metric.lower() or 'error' in metric.lower() else np.argmax(values)
                bars[best_idx].set_facecolor('gold')
                bars[best_idx].set_edgecolor('darkgoldenrod')
                bars[best_idx].set_linewidth(2)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_scalability_analysis(
        self,
        scalability_data: Dict[str, Dict[str, Dict[str, float]]],
        title: str = "Scalability Analysis",
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot scalability analysis results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Extract model names
        model_names = list(scalability_data.keys())
        
        # Plot 1: Forward pass time vs sequence length
        ax = axes[0]
        for i, model_name in enumerate(model_names):
            model_data = scalability_data[model_name]
            
            seq_lengths = []
            forward_times = []
            
            for config_name, config_data in model_data.items():
                if 'sequence_length' in config_data and 'avg_forward_time' in config_data:
                    seq_lengths.append(config_data['sequence_length'])
                    forward_times.append(config_data['avg_forward_time'] * 1000)  # Convert to ms
            
            if seq_lengths:
                # Sort by sequence length
                sorted_data = sorted(zip(seq_lengths, forward_times))
                seq_lengths, forward_times = zip(*sorted_data)
                
                ax.plot(seq_lengths, forward_times, 
                       marker='o', label=model_name, 
                       color=self.colors[i % len(self.colors)],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Forward Pass Time (ms)')
        ax.set_title('Forward Pass Time vs Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2: Throughput vs input size
        ax = axes[1]
        for i, model_name in enumerate(model_names):
            model_data = scalability_data[model_name]
            
            input_sizes = []
            throughputs = []
            
            for config_name, config_data in model_data.items():
                if 'input_size' in config_data and 'throughput' in config_data:
                    input_sizes.append(config_data['input_size'])
                    throughputs.append(config_data['throughput'])
            
            if input_sizes:
                # Sort by input size
                sorted_data = sorted(zip(input_sizes, throughputs))
                input_sizes, throughputs = zip(*sorted_data)
                
                ax.plot(input_sizes, throughputs, 
                       marker='s', label=model_name,
                       color=self.colors[i % len(self.colors)],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Throughput (sequences/sec)')
        ax.set_title('Throughput vs Input Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Memory efficiency (placeholder)
        ax = axes[2]
        ax.text(0.5, 0.5, 'Memory Efficiency\\n(To be implemented)', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, style='italic')
        ax.set_title('Memory Efficiency Analysis')
        
        # Plot 4: Computational complexity
        ax = axes[3]
        for i, model_name in enumerate(model_names):
            model_data = scalability_data[model_name]
            
            complexities = []
            times = []
            
            for config_name, config_data in model_data.items():
                # Approximate complexity as input_size * sequence_length
                if 'input_size' in config_data and 'sequence_length' in config_data:
                    complexity = config_data['input_size'] * config_data['sequence_length']
                    time = config_data.get('avg_forward_time', 0) * 1000
                    
                    complexities.append(complexity)
                    times.append(time)
            
            if complexities:
                ax.scatter(complexities, times, 
                          label=model_name, s=50,
                          color=self.colors[i % len(self.colors)],
                          alpha=0.7)
        
        ax.set_xlabel('Problem Size (Input Dim × Seq Length)')
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computational Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_statistical_significance(
        self,
        statistical_results: Dict[str, Any],
        title: str = "Statistical Significance Analysis",
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot statistical significance test results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: P-value heatmap
        ax = axes[0]
        
        if 'pairwise_tests' in statistical_results:
            pairwise_tests = statistical_results['pairwise_tests']
            model_names = list(pairwise_tests.keys())
            n_models = len(model_names)
            
            # Create p-value matrix
            p_matrix = np.ones((n_models, n_models))
            
            for i, model_a in enumerate(model_names):
                for j, model_b in enumerate(model_names):
                    if model_b in pairwise_tests[model_a]:
                        test_result = pairwise_tests[model_a][model_b]
                        if 'p_value' in test_result:
                            p_matrix[i, j] = test_result['p_value']
            
            # Create heatmap
            im = ax.imshow(p_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.1)
            
            # Add text annotations
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                     ha="center", va="center", 
                                     color="white" if p_matrix[i, j] < 0.05 else "black",
                                     fontweight='bold' if p_matrix[i, j] < 0.05 else 'normal')
            
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_yticklabels(model_names)
            ax.set_title('P-values (Pairwise t-tests)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('p-value')
            
            # Add significance threshold line
            ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Effect sizes
        ax = axes[1]
        
        if 'performance_summary' in statistical_results:
            perf_summary = statistical_results['performance_summary']
            model_names = list(perf_summary.keys())
            
            means = [perf_summary[model]['mean_performance'] for model in model_names]
            stds = [perf_summary[model]['std_performance'] for model in model_names]
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(model_names)), means, yerr=stds,
                         capsize=5, color=self.colors[:len(model_names)],
                         alpha=0.8, edgecolor='black')
            
            # Add confidence intervals
            for i, model in enumerate(model_names):
                ci = perf_summary[model].get('confidence_interval_95', [means[i], means[i]])
                ax.plot([i, i], ci, color='red', linewidth=3, alpha=0.7)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Mean Performance')
            ax.set_title('Performance with 95% Confidence Intervals')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_publication_figure(
        self,
        data: Dict[str, Any],
        figure_type: str = "comprehensive",
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create publication-ready comprehensive figures.
        """
        if figure_type == "comprehensive":
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Main performance comparison
            ax_main = fig.add_subplot(gs[0, :2])
            # Implementation details...
            
            # Training curves
            ax_training = fig.add_subplot(gs[1, :2])
            # Implementation details...
            
            # Statistical analysis
            ax_stats = fig.add_subplot(gs[2, :2])
            # Implementation details...
            
            # Scalability analysis
            ax_scale = fig.add_subplot(gs[:, 2])
            # Implementation details...
            
            plt.suptitle("Comprehensive Liquid Neural Network Analysis", 
                        fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


class NetworkVisualizer:
    """
    Visualize neural network architectures and internal dynamics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        
        if not HAS_MATPLOTLIB:
            raise VisualizationError("Matplotlib required for network visualization")
    
    def plot_network_architecture(
        self,
        model,
        title: str = "Liquid Neural Network Architecture",
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Visualize the network architecture.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get model dimensions
        if hasattr(model, 'input_size'):
            input_size = model.input_size
            hidden_size = model.hidden_size
            output_size = model.output_size
        else:
            # Default values for display
            input_size, hidden_size, output_size = 3, 8, 2
        
        # Layer positions
        layer_positions = [0, 2, 4]  # x-coordinates
        layer_sizes = [input_size, hidden_size, output_size]
        layer_labels = ['Input', 'Hidden (Liquid)', 'Output']
        
        # Draw layers
        for i, (x_pos, size, label) in enumerate(zip(layer_positions, layer_sizes, layer_labels)):
            # Draw neurons
            y_positions = np.linspace(-size/2, size/2, size)
            
            for j, y_pos in enumerate(y_positions):
                if i == 1:  # Hidden layer - different styling
                    circle = plt.Circle((x_pos, y_pos), 0.15, 
                                      color='lightblue', ec='blue', linewidth=2)
                else:
                    circle = plt.Circle((x_pos, y_pos), 0.15, 
                                      color='lightgray', ec='black', linewidth=1)
                ax.add_patch(circle)
            
            # Layer labels
            ax.text(x_pos, -size/2 - 0.8, label, ha='center', fontweight='bold')
        
        # Draw connections
        for i in range(len(layer_positions) - 1):
            x_from, x_to = layer_positions[i], layer_positions[i+1]
            size_from, size_to = layer_sizes[i], layer_sizes[i+1]
            
            y_from = np.linspace(-size_from/2, size_from/2, size_from)
            y_to = np.linspace(-size_to/2, size_to/2, size_to)
            
            # Draw subset of connections to avoid clutter
            max_connections = 20
            step_from = max(1, len(y_from) // max_connections)
            step_to = max(1, len(y_to) // max_connections)
            
            for y_f in y_from[::step_from]:
                for y_t in y_to[::step_to]:
                    ax.plot([x_from + 0.15, x_to - 0.15], [y_f, y_t], 
                           'k-', alpha=0.3, linewidth=0.5)
        
        # Special highlighting for liquid connections (recurrent)
        if len(layer_positions) > 1:
            x_pos = layer_positions[1]
            size = layer_sizes[1]
            y_positions = np.linspace(-size/2, size/2, size)
            
            # Draw some recurrent connections
            for i in range(0, len(y_positions), 2):
                for j in range(1, len(y_positions), 3):
                    if i != j:
                        y1, y2 = y_positions[i], y_positions[j]
                        # Curved recurrent connection
                        ax.annotate('', xy=(x_pos + 0.15, y2), xytext=(x_pos + 0.15, y1),
                                  arrowprops=dict(arrowstyle='->', color='red', alpha=0.6,
                                                connectionstyle="arc3,rad=0.3", lw=1))
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-max(layer_sizes)/2 - 1, max(layer_sizes)/2 + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                      markersize=10, label='Standard Neuron'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Liquid Neuron'),
            plt.Line2D([0], [0], color='black', alpha=0.3, label='Feedforward Connection'),
            plt.Line2D([0], [0], color='red', alpha=0.6, label='Recurrent Connection')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_constants_distribution(
        self,
        time_constants: Union[np.ndarray, List[float]],
        title: str = "Time Constants Distribution",
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Visualize the distribution of time constants in the network.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Convert to numpy array
        if HAS_JAX and hasattr(time_constants, 'shape'):
            tau_values = np.array(time_constants)
        else:
            tau_values = np.array(time_constants)
        
        # Histogram
        ax = axes[0]
        ax.hist(tau_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(tau_values), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(tau_values):.3f}')
        ax.axvline(np.median(tau_values), color='green', linestyle='--',
                  label=f'Median: {np.median(tau_values):.3f}')
        ax.set_xlabel('Time Constant (τ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Time Constants Histogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot with individual points
        ax = axes[1]
        ax.boxplot(tau_values, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7))
        
        # Add individual points
        y_jitter = np.random.normal(1, 0.02, len(tau_values))
        ax.scatter(y_jitter, tau_values, alpha=0.5, s=20, color='red')
        
        ax.set_ylabel('Time Constant (τ)')
        ax.set_title('Time Constants Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Statistics text
        stats_text = f"""
        Statistics:
        Mean: {np.mean(tau_values):.3f}
        Std:  {np.std(tau_values):.3f}
        Min:  {np.min(tau_values):.3f}
        Max:  {np.max(tau_values):.3f}
        """
        ax.text(1.1, 0.5, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
               verticalalignment='center')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_network_dynamics(
        self,
        states_sequence: np.ndarray,
        title: str = "Network Dynamics Animation",
        save_path: Optional[str] = None,
        interval: int = 100
    ) -> FuncAnimation:
        """
        Create animation of network state evolution over time.
        """
        if not HAS_MATPLOTLIB:
            raise VisualizationError("Matplotlib required for animations")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Setup the plot
        n_neurons = states_sequence.shape[1]
        positions = np.random.random((n_neurons, 2)) * 10  # Random positions
        
        # Initialize scatter plot
        scat = ax.scatter(positions[:, 0], positions[:, 1], 
                         s=100, c=states_sequence[0], 
                         cmap='RdBu_r', vmin=-2, vmax=2)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(scat, ax=ax)
        cbar.set_label('Activation')
        
        def animate(frame):
            """Animation function."""
            if frame < len(states_sequence):
                scat.set_array(states_sequence[frame])
            return scat,
        
        anim = FuncAnimation(fig, animate, frames=len(states_sequence),
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=10)
            else:
                anim.save(save_path, writer='ffmpeg', fps=10)
        
        return anim


def create_comprehensive_visualization_report(
    results: Dict[str, Any],
    output_dir: str = "visualization_output"
) -> Dict[str, str]:
    """
    Create a comprehensive visualization report with all plots.
    
    Args:
        results: Dictionary containing all experimental results
        output_dir: Directory to save all visualizations
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not HAS_MATPLOTLIB:
        print("Warning: Matplotlib not available. Skipping visualizations.")
        return {}
    
    visualizer = ResultsVisualizer(style='publication')
    saved_plots = {}
    
    try:
        # Training curves
        if 'training_histories' in results:
            fig = visualizer.plot_training_curves(
                results['training_histories'],
                title="Training Performance Comparison"
            )
            save_path = str(output_path / "training_curves.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_plots['training_curves'] = save_path
        
        # Performance comparison
        if 'performance_results' in results:
            fig = visualizer.plot_performance_comparison(
                results['performance_results'],
                metrics=['final_loss', 'training_time', 'convergence_rate'],
                title="Model Performance Comparison"
            )
            save_path = str(output_path / "performance_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_plots['performance_comparison'] = save_path
        
        # Scalability analysis
        if 'scalability_results' in results:
            fig = visualizer.plot_scalability_analysis(
                results['scalability_results'],
                title="Computational Scalability Analysis"
            )
            save_path = str(output_path / "scalability_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_plots['scalability_analysis'] = save_path
        
        # Statistical significance
        if 'statistical_results' in results:
            fig = visualizer.plot_statistical_significance(
                results['statistical_results'],
                title="Statistical Significance Analysis"
            )
            save_path = str(output_path / "statistical_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_plots['statistical_analysis'] = save_path
        
        print(f"Visualization report generated in {output_dir}")
        print(f"Created {len(saved_plots)} plots:")
        for plot_name, file_path in saved_plots.items():
            print(f"  - {plot_name}: {file_path}")
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return saved_plots