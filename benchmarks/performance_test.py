"""
Performance benchmarks for temporal fairness metrics.

Measures execution time and memory usage for all implemented metrics
with various dataset sizes and configurations.
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tabulate import tabulate

# Import metrics
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.utils.data_generators import TemporalBiasGenerator


class PerformanceBenchmark:
    """Benchmark suite for temporal fairness metrics."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = []
        self.generator = TemporalBiasGenerator(random_seed=42)
    
    def benchmark_metric(
        self,
        metric_name: str,
        metric_func: callable,
        data: Dict[str, np.ndarray],
        n_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark a single metric.
        
        Args:
            metric_name: Name of the metric
            metric_func: Function to benchmark
            data: Input data for the metric
            n_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with performance statistics
        """
        execution_times = []
        memory_usage = []
        
        for _ in range(n_iterations):
            # Measure memory
            tracemalloc.start()
            
            # Measure execution time
            start_time = time.perf_counter()
            result = metric_func(**data)
            end_time = time.perf_counter()
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_times.append(end_time - start_time)
            memory_usage.append(peak / 1024 / 1024)  # Convert to MB
        
        return {
            'metric': metric_name,
            'mean_time': np.mean(execution_times),
            'std_time': np.std(execution_times),
            'min_time': np.min(execution_times),
            'max_time': np.max(execution_times),
            'mean_memory_mb': np.mean(memory_usage),
            'n_samples': len(data.get('decisions', data.get('predictions', data.get('metric_history', []))))
        }
    
    def run_tdp_benchmarks(self) -> List[Dict[str, float]]:
        """Run benchmarks for Temporal Demographic Parity."""
        tdp = TemporalDemographicParity(threshold=0.1)
        results = []
        
        # Test different dataset sizes
        for n_samples in [100, 500, 1000, 5000, 10000, 50000]:
            # Generate data
            df = self.generator.generate_dataset(
                n_samples=n_samples,
                n_groups=3,
                time_periods=10
            )
            
            data = {
                'decisions': df['decision'].values,
                'groups': df['group'].values,
                'timestamps': df['time_period'].values,
                'return_details': True
            }
            
            # Run benchmark
            result = self.benchmark_metric(
                metric_name=f'TDP_{n_samples}',
                metric_func=tdp.calculate,
                data=data
            )
            result['dataset_size'] = n_samples
            results.append(result)
            
            print(f"TDP with {n_samples:,} samples: {result['mean_time']*1000:.2f}ms")
        
        return results
    
    def run_eoot_benchmarks(self) -> List[Dict[str, float]]:
        """Run benchmarks for Equalized Odds Over Time."""
        eoot = EqualizedOddsOverTime(tpr_threshold=0.1, fpr_threshold=0.1)
        results = []
        
        # Test different dataset sizes
        for n_samples in [100, 500, 1000, 5000, 10000, 50000]:
            # Generate data
            df = self.generator.generate_dataset(
                n_samples=n_samples,
                n_groups=3,
                time_periods=10
            )
            
            # Create synthetic true labels
            true_labels = np.random.binomial(1, 0.5, size=n_samples)
            
            data = {
                'predictions': df['decision'].values,
                'true_labels': true_labels,
                'groups': df['group'].values,
                'timestamps': df['time_period'].values,
                'return_details': True
            }
            
            # Run benchmark
            result = self.benchmark_metric(
                metric_name=f'EOOT_{n_samples}',
                metric_func=eoot.calculate,
                data=data
            )
            result['dataset_size'] = n_samples
            results.append(result)
            
            print(f"EOOT with {n_samples:,} samples: {result['mean_time']*1000:.2f}ms")
        
        return results
    
    def run_fdd_benchmarks(self) -> List[Dict[str, float]]:
        """Run benchmarks for Fairness Decay Detection."""
        fdd = FairnessDecayDetection(decay_threshold=0.05)
        results = []
        
        # Test different history lengths
        for history_length in [10, 50, 100, 500, 1000]:
            # Generate metric history
            metric_history = np.random.random(history_length)
            timestamps = np.arange(history_length)
            
            data = {
                'metric_history': metric_history,
                'timestamps': timestamps,
                'return_details': True
            }
            
            # Run benchmark
            result = self.benchmark_metric(
                metric_name=f'FDD_{history_length}',
                metric_func=fdd.detect_fairness_decay,
                data=data
            )
            result['history_length'] = history_length
            results.append(result)
            
            print(f"FDD with {history_length} time points: {result['mean_time']*1000:.2f}ms")
        
        return results
    
    def run_window_size_benchmark(self) -> List[Dict[str, float]]:
        """Benchmark impact of different window sizes."""
        tdp = TemporalDemographicParity()
        results = []
        
        # Fixed dataset size
        n_samples = 10000
        df = self.generator.generate_dataset(
            n_samples=n_samples,
            n_groups=3,
            time_periods=100
        )
        
        # Test different window configurations
        for n_windows in [1, 5, 10, 20, 50, 100]:
            # Create time windows
            time_range = df['time_period'].max() - df['time_period'].min()
            window_size = time_range / n_windows
            
            data = {
                'decisions': df['decision'].values,
                'groups': df['group'].values,
                'timestamps': df['time_period'].values,
                'window_size': window_size,
                'return_details': True
            }
            
            # Run benchmark
            result = self.benchmark_metric(
                metric_name=f'TDP_windows_{n_windows}',
                metric_func=tdp.calculate,
                data=data
            )
            result['n_windows'] = n_windows
            results.append(result)
            
            print(f"TDP with {n_windows} windows: {result['mean_time']*1000:.2f}ms")
        
        return results
    
    def run_group_count_benchmark(self) -> List[Dict[str, float]]:
        """Benchmark impact of number of groups."""
        tdp = TemporalDemographicParity()
        results = []
        
        # Fixed dataset size
        n_samples = 10000
        
        # Test different group counts
        for n_groups in [2, 5, 10, 20, 50]:
            df = self.generator.generate_dataset(
                n_samples=n_samples,
                n_groups=n_groups,
                time_periods=10
            )
            
            data = {
                'decisions': df['decision'].values,
                'groups': df['group'].values,
                'timestamps': df['time_period'].values,
                'return_details': True
            }
            
            # Run benchmark
            result = self.benchmark_metric(
                metric_name=f'TDP_groups_{n_groups}',
                metric_func=tdp.calculate,
                data=data
            )
            result['n_groups'] = n_groups
            results.append(result)
            
            print(f"TDP with {n_groups} groups: {result['mean_time']*1000:.2f}ms")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[Dict[str, float]]]:
        """Run all benchmarks and compile results."""
        print("\n" + "="*60)
        print("TEMPORAL FAIRNESS METRICS - PERFORMANCE BENCHMARK")
        print("="*60)
        
        all_results = {}
        
        # TDP Benchmarks
        print("\n[1/5] Running TDP benchmarks...")
        all_results['tdp'] = self.run_tdp_benchmarks()
        
        # EOOT Benchmarks
        print("\n[2/5] Running EOOT benchmarks...")
        all_results['eoot'] = self.run_eoot_benchmarks()
        
        # FDD Benchmarks
        print("\n[3/5] Running FDD benchmarks...")
        all_results['fdd'] = self.run_fdd_benchmarks()
        
        # Window size impact
        print("\n[4/5] Running window size benchmarks...")
        all_results['windows'] = self.run_window_size_benchmark()
        
        # Group count impact
        print("\n[5/5] Running group count benchmarks...")
        all_results['groups'] = self.run_group_count_benchmark()
        
        return all_results
    
    def generate_report(self, results: Dict[str, List[Dict[str, float]]]) -> str:
        """Generate performance report."""
        report = []
        report.append("\n" + "="*60)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("="*60)
        
        # TDP Performance vs Dataset Size
        if 'tdp' in results:
            report.append("\n## Temporal Demographic Parity (TDP)")
            table_data = []
            for r in results['tdp']:
                table_data.append([
                    f"{r['dataset_size']:,}",
                    f"{r['mean_time']*1000:.2f}ms",
                    f"{r['std_time']*1000:.2f}ms",
                    f"{r['mean_memory_mb']:.2f}MB"
                ])
            report.append(tabulate(
                table_data,
                headers=['Samples', 'Mean Time', 'Std Dev', 'Memory'],
                tablefmt='grid'
            ))
        
        # EOOT Performance vs Dataset Size
        if 'eoot' in results:
            report.append("\n## Equalized Odds Over Time (EOOT)")
            table_data = []
            for r in results['eoot']:
                table_data.append([
                    f"{r['dataset_size']:,}",
                    f"{r['mean_time']*1000:.2f}ms",
                    f"{r['std_time']*1000:.2f}ms",
                    f"{r['mean_memory_mb']:.2f}MB"
                ])
            report.append(tabulate(
                table_data,
                headers=['Samples', 'Mean Time', 'Std Dev', 'Memory'],
                tablefmt='grid'
            ))
        
        # FDD Performance vs History Length
        if 'fdd' in results:
            report.append("\n## Fairness Decay Detection (FDD)")
            table_data = []
            for r in results['fdd']:
                table_data.append([
                    f"{r['history_length']}",
                    f"{r['mean_time']*1000:.2f}ms",
                    f"{r['std_time']*1000:.2f}ms",
                    f"{r['mean_memory_mb']:.2f}MB"
                ])
            report.append(tabulate(
                table_data,
                headers=['History Length', 'Mean Time', 'Std Dev', 'Memory'],
                tablefmt='grid'
            ))
        
        # Performance vs Window Count
        if 'windows' in results:
            report.append("\n## Impact of Window Count (10K samples)")
            table_data = []
            for r in results['windows']:
                table_data.append([
                    f"{r['n_windows']}",
                    f"{r['mean_time']*1000:.2f}ms",
                    f"{r['std_time']*1000:.2f}ms"
                ])
            report.append(tabulate(
                table_data,
                headers=['Windows', 'Mean Time', 'Std Dev'],
                tablefmt='grid'
            ))
        
        # Performance vs Group Count
        if 'groups' in results:
            report.append("\n## Impact of Group Count (10K samples)")
            table_data = []
            for r in results['groups']:
                table_data.append([
                    f"{r['n_groups']}",
                    f"{r['mean_time']*1000:.2f}ms",
                    f"{r['std_time']*1000:.2f}ms"
                ])
            report.append(tabulate(
                table_data,
                headers=['Groups', 'Mean Time', 'Std Dev'],
                tablefmt='grid'
            ))
        
        # Key Performance Metrics
        report.append("\n## Key Performance Metrics")
        report.append("-" * 40)
        
        # Check 10K record performance requirement
        tdp_10k = next((r for r in results.get('tdp', []) if r['dataset_size'] == 10000), None)
        eoot_10k = next((r for r in results.get('eoot', []) if r['dataset_size'] == 10000), None)
        
        if tdp_10k:
            meets_req = tdp_10k['mean_time'] < 0.1
            status = "✓ PASS" if meets_req else "✗ FAIL"
            report.append(f"TDP (10K records):  {tdp_10k['mean_time']*1000:.2f}ms {status}")
        
        if eoot_10k:
            meets_req = eoot_10k['mean_time'] < 0.1
            status = "✓ PASS" if meets_req else "✗ FAIL"
            report.append(f"EOOT (10K records): {eoot_10k['mean_time']*1000:.2f}ms {status}")
        
        # FDD with 100 points
        fdd_100 = next((r for r in results.get('fdd', []) if r['history_length'] == 100), None)
        if fdd_100:
            meets_req = fdd_100['mean_time'] < 0.1
            status = "✓ PASS" if meets_req else "✗ FAIL"
            report.append(f"FDD (100 points):   {fdd_100['mean_time']*1000:.2f}ms {status}")
        
        report.append("\nRequirement: <100ms for 10K records")
        
        return "\n".join(report)
    
    def plot_results(self, results: Dict[str, List[Dict[str, float]]]):
        """Generate performance visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Temporal Fairness Metrics - Performance Analysis', fontsize=16)
        
        # TDP Performance
        if 'tdp' in results:
            ax = axes[0, 0]
            sizes = [r['dataset_size'] for r in results['tdp']]
            times = [r['mean_time']*1000 for r in results['tdp']]
            ax.plot(sizes, times, 'b-o', linewidth=2, markersize=8)
            ax.axhline(y=100, color='r', linestyle='--', label='100ms target')
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('TDP Performance')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # EOOT Performance
        if 'eoot' in results:
            ax = axes[0, 1]
            sizes = [r['dataset_size'] for r in results['eoot']]
            times = [r['mean_time']*1000 for r in results['eoot']]
            ax.plot(sizes, times, 'g-o', linewidth=2, markersize=8)
            ax.axhline(y=100, color='r', linestyle='--', label='100ms target')
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('EOOT Performance')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # FDD Performance
        if 'fdd' in results:
            ax = axes[0, 2]
            lengths = [r['history_length'] for r in results['fdd']]
            times = [r['mean_time']*1000 for r in results['fdd']]
            ax.plot(lengths, times, 'm-o', linewidth=2, markersize=8)
            ax.axhline(y=100, color='r', linestyle='--', label='100ms target')
            ax.set_xlabel('History Length')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('FDD Performance')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Memory Usage Comparison
        if all(k in results for k in ['tdp', 'eoot']):
            ax = axes[1, 0]
            sizes = [r['dataset_size'] for r in results['tdp']]
            tdp_mem = [r['mean_memory_mb'] for r in results['tdp']]
            eoot_mem = [r['mean_memory_mb'] for r in results['eoot']]
            
            ax.plot(sizes, tdp_mem, 'b-s', label='TDP', linewidth=2, markersize=8)
            ax.plot(sizes, eoot_mem, 'g-^', label='EOOT', linewidth=2, markersize=8)
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Consumption')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Window Count Impact
        if 'windows' in results:
            ax = axes[1, 1]
            windows = [r['n_windows'] for r in results['windows']]
            times = [r['mean_time']*1000 for r in results['windows']]
            ax.plot(windows, times, 'c-o', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Windows')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('Impact of Window Count')
            ax.grid(True, alpha=0.3)
        
        # Group Count Impact
        if 'groups' in results:
            ax = axes[1, 2]
            groups = [r['n_groups'] for r in results['groups']]
            times = [r['mean_time']*1000 for r in results['groups']]
            ax.plot(groups, times, 'y-o', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Groups')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('Impact of Group Count')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=150)
        plt.show()


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate report
    report = benchmark.generate_report(results)
    print(report)
    
    # Save report to file
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)
    
    # Generate plots
    try:
        benchmark.plot_results(results)
        print("\nPerformance plots saved to 'benchmark_results.png'")
    except ImportError:
        print("\nMatplotlib not available for plotting")
    
    print("\nBenchmark report saved to 'benchmark_report.txt'")


if __name__ == "__main__":
    main()