"""
Visualization module for temporal fairness metrics.

Provides comprehensive visualization tools for analyzing temporal fairness
patterns, trends, and metric evolution over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FairnessVisualizer:
    """Visualization tools for temporal fairness analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
    
    def plot_metric_evolution(
        self,
        metric_values: Union[List[float], np.ndarray, pd.Series],
        timestamps: Optional[Union[List, np.ndarray, pd.Series]] = None,
        metric_name: str = 'Fairness Metric',
        threshold: Optional[float] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the evolution of a fairness metric over time.
        
        Args:
            metric_values: Values of the metric over time
            timestamps: Timestamps for each value
            metric_name: Name of the metric
            threshold: Fairness threshold to display
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if timestamps is None:
            timestamps = np.arange(len(metric_values))
        
        # Main metric line
        ax.plot(timestamps, metric_values, 'b-', linewidth=2, label=metric_name)
        ax.scatter(timestamps, metric_values, c='blue', s=50, alpha=0.6)
        
        # Add threshold line
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      linewidth=1.5, label=f'Threshold ({threshold:.2f})')
            
            # Shade violation regions
            violations = np.array(metric_values) > threshold
            if np.any(violations):
                ax.fill_between(timestamps, 0, 1, where=violations,
                               color='red', alpha=0.1, transform=ax.get_xaxis_transform(),
                               label='Violation Period')
        
        # Add trend line
        z = np.polyfit(range(len(metric_values)), metric_values, 1)
        p = np.poly1d(z)
        ax.plot(timestamps, p(range(len(metric_values))), 
               'g--', alpha=0.5, label='Trend')
        
        # Formatting
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title or f'{metric_name} Evolution Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'Mean: {np.mean(metric_values):.3f}\n'
        stats_text += f'Std: {np.std(metric_values):.3f}\n'
        stats_text += f'Min: {np.min(metric_values):.3f}\n'
        stats_text += f'Max: {np.max(metric_values):.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_group_comparison(
        self,
        data: pd.DataFrame,
        metric_col: str,
        group_col: str,
        time_col: Optional[str] = None,
        plot_type: str = 'line',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare metric values across different groups.
        
        Args:
            data: DataFrame with metric data
            metric_col: Column name for metric values
            group_col: Column name for groups
            time_col: Column name for time (optional)
            plot_type: Type of plot ('line', 'box', 'violin', 'bar')
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if plot_type == 'line' and time_col:
            # Line plot over time for each group
            for i, group in enumerate(data[group_col].unique()):
                group_data = data[data[group_col] == group].sort_values(time_col)
                ax.plot(group_data[time_col], group_data[metric_col],
                       marker='o', label=group, color=self.color_palette[i % len(self.color_palette)])
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(metric_col, fontsize=12)
            
        elif plot_type == 'box':
            # Box plot comparison
            groups = []
            values = []
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]
                groups.extend([group] * len(group_data))
                values.extend(group_data[metric_col].values)
            
            box_data = pd.DataFrame({'Group': groups, 'Value': values})
            sns.boxplot(data=box_data, x='Group', y='Value', ax=ax)
            ax.set_ylabel(metric_col, fontsize=12)
            
        elif plot_type == 'violin':
            # Violin plot comparison
            groups = []
            values = []
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]
                groups.extend([group] * len(group_data))
                values.extend(group_data[metric_col].values)
            
            violin_data = pd.DataFrame({'Group': groups, 'Value': values})
            sns.violinplot(data=violin_data, x='Group', y='Value', ax=ax)
            ax.set_ylabel(metric_col, fontsize=12)
            
        elif plot_type == 'bar':
            # Bar plot of means
            group_means = data.groupby(group_col)[metric_col].mean()
            group_stds = data.groupby(group_col)[metric_col].std()
            
            x_pos = np.arange(len(group_means))
            ax.bar(x_pos, group_means.values, yerr=group_stds.values,
                  capsize=5, color=self.color_palette[:len(group_means)])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_means.index)
            ax.set_ylabel(f'Mean {metric_col}', fontsize=12)
        
        ax.set_title(title or f'{metric_col} by {group_col}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_fairness_heatmap(
        self,
        metrics_matrix: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = 'Fairness Metrics Heatmap',
        cmap: str = 'RdYlGn_r',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap of fairness metrics.
        
        Args:
            metrics_matrix: 2D array of metric values
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Plot title
            cmap: Colormap name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(metrics_matrix, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        if row_labels:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
        
        if col_labels:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Metric Value', rotation=270, labelpad=20)
        
        # Add value annotations
        for i in range(metrics_matrix.shape[0]):
            for j in range(metrics_matrix.shape[1]):
                text = ax.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_decay_analysis(
        self,
        metric_history: List[float],
        decay_info: Dict[str, Any],
        predictions: Optional[Dict[str, Any]] = None,
        title: str = 'Fairness Decay Analysis',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize fairness decay detection results.
        
        Args:
            metric_history: Historical metric values
            decay_info: Decay detection results
            predictions: Future predictions (optional)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Plot 1: Metric history with decay trend
        time_points = np.arange(len(metric_history))
        ax1.plot(time_points, metric_history, 'b-', linewidth=2, label='Actual')
        ax1.scatter(time_points, metric_history, c='blue', s=50, alpha=0.6)
        
        # Add decay trend if detected
        if decay_info.get('decay_detected'):
            if 'slope' in decay_info:
                # Linear trend
                z = np.polyfit(time_points, metric_history, 1)
                p = np.poly1d(z)
                ax1.plot(time_points, p(time_points), 'r--', 
                        linewidth=2, label=f'Decay Trend (slope={decay_info["slope"]:.4f})')
            
            if 'changepoint_index' in decay_info and decay_info['changepoint_index']:
                # Mark changepoint
                cp_idx = decay_info['changepoint_index']
                ax1.axvline(x=cp_idx, color='orange', linestyle='--', 
                           linewidth=2, label=f'Changepoint')
        
        # Add predictions if available
        if predictions and predictions.get('predictions') is not None:
            future_points = np.arange(len(metric_history), 
                                     len(metric_history) + len(predictions['predictions']))
            ax1.plot(future_points, predictions['predictions'], 
                    'g--', linewidth=2, label='Predicted')
            
            if 'confidence_lower' in predictions:
                ax1.fill_between(future_points,
                                predictions['confidence_lower'],
                                predictions['confidence_upper'],
                                color='green', alpha=0.2, label='95% CI')
        
        ax1.set_xlabel('Time Period', fontsize=12)
        ax1.set_ylabel('Metric Value', fontsize=12)
        ax1.set_title('Metric Evolution and Decay Detection', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rate of change
        if len(metric_history) > 1:
            changes = np.diff(metric_history)
            ax2.bar(range(len(changes)), changes, 
                   color=['red' if c < 0 else 'green' for c in changes],
                   alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Time Period', fontsize=12)
            ax2.set_ylabel('Period-to-Period Change', fontsize=12)
            ax2.set_title('Rate of Change Analysis', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_queue_fairness(
        self,
        queue_data: pd.DataFrame,
        position_col: str = 'queue_position',
        group_col: str = 'group',
        wait_col: Optional[str] = None,
        title: str = 'Queue Position Fairness Analysis',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize queue position fairness.
        
        Args:
            queue_data: DataFrame with queue data
            position_col: Column name for queue positions
            group_col: Column name for groups
            wait_col: Column name for wait times (optional)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_plots = 3 if wait_col else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 0.6))
        
        if n_plots == 2:
            axes = list(axes) + [None]
        
        # Plot 1: Queue position distribution by group
        groups = []
        positions = []
        for group in queue_data[group_col].unique():
            group_data = queue_data[queue_data[group_col] == group]
            groups.extend([group] * len(group_data))
            positions.extend(group_data[position_col].values)
        
        position_df = pd.DataFrame({'Group': groups, 'Position': positions})
        sns.violinplot(data=position_df, x='Group', y='Position', ax=axes[0])
        axes[0].set_title('Queue Position Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Queue Position (lower = better)', fontsize=11)
        axes[0].invert_yaxis()  # Lower positions are better
        
        # Plot 2: Average position by group
        avg_positions = queue_data.groupby(group_col)[position_col].mean()
        std_positions = queue_data.groupby(group_col)[position_col].std()
        
        x_pos = np.arange(len(avg_positions))
        axes[1].bar(x_pos, avg_positions.values, yerr=std_positions.values,
                   capsize=5, color=self.color_palette[:len(avg_positions)])
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(avg_positions.index)
        axes[1].set_title('Average Queue Position', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Position', fontsize=11)
        
        # Plot 3: Wait time distribution (if available)
        if wait_col and axes[2] is not None:
            wait_groups = []
            wait_times = []
            for group in queue_data[group_col].unique():
                group_data = queue_data[queue_data[group_col] == group]
                wait_groups.extend([group] * len(group_data))
                wait_times.extend(group_data[wait_col].values)
            
            wait_df = pd.DataFrame({'Group': wait_groups, 'Wait': wait_times})
            sns.boxplot(data=wait_df, x='Group', y='Wait', ax=axes[2])
            axes[2].set_title('Wait Time Distribution', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Wait Time', fontsize=11)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(
        self,
        metrics_data: Dict[str, Any],
        title: str = 'Temporal Fairness Dashboard',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple metrics.
        
        Args:
            metrics_data: Dictionary with metric names and values
            title: Dashboard title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Define subplot positions
        ax1 = fig.add_subplot(gs[0, :2])  # Top left - main metric evolution
        ax2 = fig.add_subplot(gs[0, 2])   # Top right - summary stats
        ax3 = fig.add_subplot(gs[1, 0])   # Middle left - group comparison
        ax4 = fig.add_subplot(gs[1, 1])   # Middle center - heatmap
        ax5 = fig.add_subplot(gs[1, 2])   # Middle right - distribution
        ax6 = fig.add_subplot(gs[2, :])   # Bottom - detailed timeline
        
        # Plot 1: Main metric evolution
        if 'tdp_history' in metrics_data:
            tdp_vals = metrics_data['tdp_history']
            ax1.plot(range(len(tdp_vals)), tdp_vals, 'b-', linewidth=2, label='TDP')
            ax1.fill_between(range(len(tdp_vals)), 0, tdp_vals, alpha=0.3)
        
        if 'eoot_history' in metrics_data:
            eoot_vals = metrics_data['eoot_history']
            ax1.plot(range(len(eoot_vals)), eoot_vals, 'g-', linewidth=2, label='EOOT')
        
        ax1.set_title('Metric Evolution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Metric Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Summary statistics table
        if 'summary_stats' in metrics_data:
            stats = metrics_data['summary_stats']
            table_data = [[k, f'{v:.3f}'] for k, v in stats.items()]
            ax2.axis('tight')
            ax2.axis('off')
            table = ax2.table(cellText=table_data, colLabels=['Metric', 'Value'],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax2.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        # Plot 3: Group comparison
        if 'group_metrics' in metrics_data:
            group_data = metrics_data['group_metrics']
            groups = list(group_data.keys())
            values = list(group_data.values())
            ax3.bar(groups, values, color=self.color_palette[:len(groups)])
            ax3.set_title('Metrics by Group', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Metric Value')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Correlation heatmap
        if 'correlation_matrix' in metrics_data:
            corr_matrix = metrics_data['correlation_matrix']
            im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_title('Metric Correlations', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # Plot 5: Distribution
        if 'metric_distribution' in metrics_data:
            dist_data = metrics_data['metric_distribution']
            ax5.hist(dist_data, bins=20, edgecolor='black', alpha=0.7)
            ax5.set_title('Value Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Metric Value')
            ax5.set_ylabel('Frequency')
        
        # Plot 6: Detailed timeline with annotations
        if 'timeline_events' in metrics_data:
            events = metrics_data['timeline_events']
            for i, event in enumerate(events):
                ax6.scatter(event['time'], event['value'], s=100, 
                          c=self.color_palette[i % len(self.color_palette)])
                ax6.annotate(event['label'], (event['time'], event['value']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            ax6.set_title('Event Timeline', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Metric Value')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def create_fairness_report(
    metrics_results: Dict[str, Any],
    output_path: str = 'fairness_report.png'
) -> None:
    """
    Create a comprehensive fairness report visualization.
    
    Args:
        metrics_results: Dictionary with all metric results
        output_path: Path to save the report
    """
    visualizer = FairnessVisualizer(figsize=(16, 12))
    
    # Prepare data for dashboard
    dashboard_data = {
        'tdp_history': metrics_results.get('tdp_values', []),
        'eoot_history': metrics_results.get('eoot_values', []),
        'summary_stats': {
            'Mean TDP': np.mean(metrics_results.get('tdp_values', [0])),
            'Mean EOOT': np.mean(metrics_results.get('eoot_values', [0])),
            'QPF Score': metrics_results.get('qpf_score', 0),
            'Decay Detected': int(metrics_results.get('decay_detected', False))
        },
        'group_metrics': metrics_results.get('group_metrics', {}),
        'metric_distribution': metrics_results.get('all_metric_values', []),
        'timeline_events': metrics_results.get('events', [])
    }
    
    # Create and save dashboard
    fig = visualizer.create_dashboard(dashboard_data, title='Temporal Fairness Analysis Report')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Fairness report saved to {output_path}")