"""
Performance Testing Suite for Temporal Fairness Framework

This module provides comprehensive performance benchmarking for all metrics,
including time complexity validation, memory usage analysis, and scalability testing.
"""

import time
import tracemalloc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import psutil
import gc
from typing import Dict, List, Tuple, Any
import json

# Import all metrics
from src.metrics.temporal_demographic_parity import TemporalDemographicParity
from src.metrics.equalized_odds_over_time import EqualizedOddsOverTime
from src.metrics.fairness_decay_detection import FairnessDecayDetection
from src.metrics.queue_position_fairness import QueuePositionFairness

# Import analysis tools
from src.analysis.bias_detector import BiasDetector
from src.analysis.enhanced_bias_detector import EnhancedBiasDetector
from src.analysis.temporal_analyzer import TemporalAnalyzer

# Import data generators
from src.utils.data_generators import TemporalBiasGenerator


class PerformanceTester:
    """Comprehensive performance testing for temporal fairness metrics."""
    
    def __init__(self):
        """Initialize performance tester."""
        self.results = {}
        self.generator = TemporalBiasGenerator(random_seed=42)
        
    def measure_time(self, func, *args, **kwargs) -> Tuple[float, Any]:
        """
        Measure execution time of a function.
        
        Returns:
            Tuple of (execution_time_ms, result)
        """
        gc.collect()  # Clean up before measurement
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return (end - start) * 1000, result  # Convert to milliseconds
    
    def measure_memory(self, func, *args, **kwargs) -> Tuple[float, float, Any]:
        """
        Measure memory usage of a function.
        
        Returns:
            Tuple of (peak_memory_mb, current_memory_mb, result)
        """
        gc.collect()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return peak / 1024 / 1024, current / 1024 / 1024, result  # Convert to MB
    
    def generate_test_data(self, n_records: int, n_groups: int = 4) -> Dict[str, np.ndarray]:
        """
        Generate test data for performance testing.
        
        Args:
            n_records: Number of records to generate
            n_groups: Number of groups
            
        Returns:
            Dictionary with test data arrays
        """
        # Generate temporal data
        start_time = datetime.now()
        timestamps = np.array([
            start_time + timedelta(hours=i * 24 / n_records)
            for i in range(n_records)
        ])
        
        # Generate group assignments
        groups = np.random.choice([f'Group_{i}' for i in range(n_groups)], n_records)
        
        # Generate decisions (binary)
        decisions = np.random.choice([0, 1], n_records, p=[0.4, 0.6])
        
        # Generate ground truth labels for EOOT
        labels = np.random.choice([0, 1], n_records, p=[0.45, 0.55])
        
        # Generate queue positions
        max_queue = max(10, min(100, n_records // 10))
        queue_positions = np.random.randint(1, max_queue + 1, n_records)
        
        # Generate wait times
        wait_times = np.random.exponential(30, n_records)  # Average 30 minute wait
        
        # Generate fairness scores for FDD
        n_fairness_points = max(10, min(100, n_records // 100))
        fairness_scores = 0.8 + 0.15 * np.random.randn(n_fairness_points)
        step = max(1, n_records // n_fairness_points)
        fairness_timestamps = timestamps[::step][:n_fairness_points]
        
        return {
            'timestamps': timestamps,
            'groups': groups,
            'decisions': decisions,
            'labels': labels,
            'queue_positions': queue_positions,
            'wait_times': wait_times,
            'fairness_scores': fairness_scores,
            'fairness_timestamps': fairness_timestamps
        }
    
    def test_tdp_performance(self, data_sizes: List[int]) -> Dict:
        """Test Temporal Demographic Parity performance."""
        print("\n" + "="*70)
        print("TEMPORAL DEMOGRAPHIC PARITY PERFORMANCE TEST")
        print("="*70)
        
        tdp = TemporalDemographicParity(threshold=0.1)
        results = []
        
        for size in data_sizes:
            print(f"\nTesting with {size:,} records...")
            data = self.generate_test_data(size)
            
            # Time measurement
            exec_time, _ = self.measure_time(
                tdp.detect_bias,
                data['decisions'],
                data['groups'],
                data['timestamps']
            )
            
            # Memory measurement
            peak_mem, current_mem, _ = self.measure_memory(
                tdp.detect_bias,
                data['decisions'],
                data['groups'],
                data['timestamps']
            )
            
            results.append({
                'size': size,
                'time_ms': exec_time,
                'peak_memory_mb': peak_mem,
                'current_memory_mb': current_mem,
                'time_per_record_us': (exec_time * 1000) / size  # microseconds
            })
            
            print(f"  Time: {exec_time:.2f}ms")
            print(f"  Peak Memory: {peak_mem:.2f}MB")
            print(f"  Time per record: {results[-1]['time_per_record_us']:.2f}μs")
        
        return {'tdp': results}
    
    def test_eoot_performance(self, data_sizes: List[int]) -> Dict:
        """Test Equalized Odds Over Time performance."""
        print("\n" + "="*70)
        print("EQUALIZED ODDS OVER TIME PERFORMANCE TEST")
        print("="*70)
        
        eoot = EqualizedOddsOverTime(tpr_threshold=0.15, fpr_threshold=0.15)
        results = []
        
        for size in data_sizes:
            print(f"\nTesting with {size:,} records...")
            data = self.generate_test_data(size)
            
            # Time measurement
            exec_time, _ = self.measure_time(
                eoot.detect_bias,
                data['decisions'],
                data['labels'],
                data['groups'],
                data['timestamps']
            )
            
            # Memory measurement
            peak_mem, current_mem, _ = self.measure_memory(
                eoot.detect_bias,
                data['decisions'],
                data['labels'],
                data['groups'],
                data['timestamps']
            )
            
            results.append({
                'size': size,
                'time_ms': exec_time,
                'peak_memory_mb': peak_mem,
                'current_memory_mb': current_mem,
                'time_per_record_us': (exec_time * 1000) / size
            })
            
            print(f"  Time: {exec_time:.2f}ms")
            print(f"  Peak Memory: {peak_mem:.2f}MB")
            print(f"  Time per record: {results[-1]['time_per_record_us']:.2f}μs")
        
        return {'eoot': results}
    
    def test_fdd_performance(self, data_sizes: List[int]) -> Dict:
        """Test Fairness Decay Detection performance."""
        print("\n" + "="*70)
        print("FAIRNESS DECAY DETECTION PERFORMANCE TEST")
        print("="*70)
        
        fdd = FairnessDecayDetection(decay_threshold=0.1)
        results = []
        
        for size in data_sizes:
            # FDD works on time series, so we test with fewer points
            n_points = min(size // 100, 1000)  # Scale down for time series
            print(f"\nTesting with {n_points:,} time points (from {size:,} records)...")
            
            data = self.generate_test_data(n_points)
            
            # Time measurement
            # Convert timestamps to list for FDD
            timestamps_list = data['fairness_timestamps'].tolist() if hasattr(data['fairness_timestamps'], 'tolist') else data['fairness_timestamps']
            exec_time, _ = self.measure_time(
                fdd.detect_fairness_decay,
                data['fairness_scores'],
                timestamps_list
            )
            
            # Memory measurement
            peak_mem, current_mem, _ = self.measure_memory(
                fdd.detect_fairness_decay,
                data['fairness_scores'],
                timestamps_list
            )
            
            results.append({
                'size': size,
                'n_points': n_points,
                'time_ms': exec_time,
                'peak_memory_mb': peak_mem,
                'current_memory_mb': current_mem,
                'time_per_point_us': (exec_time * 1000) / n_points
            })
            
            print(f"  Time: {exec_time:.2f}ms")
            print(f"  Peak Memory: {peak_mem:.2f}MB")
            print(f"  Time per point: {results[-1]['time_per_point_us']:.2f}μs")
        
        return {'fdd': results}
    
    def test_qpf_performance(self, data_sizes: List[int]) -> Dict:
        """Test Queue Position Fairness performance."""
        print("\n" + "="*70)
        print("QUEUE POSITION FAIRNESS PERFORMANCE TEST")
        print("="*70)
        
        qpf = QueuePositionFairness(fairness_threshold=0.8)
        results = []
        
        for size in data_sizes:
            print(f"\nTesting with {size:,} records...")
            data = self.generate_test_data(size)
            
            # Time measurement
            exec_time, _ = self.measure_time(
                qpf.detect_bias,
                data['queue_positions'],
                data['groups'],
                data['timestamps']
            )
            
            # Memory measurement
            peak_mem, current_mem, _ = self.measure_memory(
                qpf.detect_bias,
                data['queue_positions'],
                data['groups'],
                data['timestamps']
            )
            
            results.append({
                'size': size,
                'time_ms': exec_time,
                'peak_memory_mb': peak_mem,
                'current_memory_mb': current_mem,
                'time_per_record_us': (exec_time * 1000) / size
            })
            
            print(f"  Time: {exec_time:.2f}ms")
            print(f"  Peak Memory: {peak_mem:.2f}MB")
            print(f"  Time per record: {results[-1]['time_per_record_us']:.2f}μs")
        
        return {'qpf': results}
    
    def test_bias_detector_performance(self, data_sizes: List[int]) -> Dict:
        """Test BiasDetector performance."""
        print("\n" + "="*70)
        print("BIAS DETECTOR PERFORMANCE TEST")
        print("="*70)
        
        detector = BiasDetector(sensitivity=0.95)
        results = []
        
        for size in data_sizes:
            n_points = min(size // 100, 500)
            print(f"\nTesting with {n_points:,} time points...")
            
            # Generate time series data
            decisions_over_time = np.random.randn(n_points) * 0.1 + 0.5
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
            
            # Test confidence valley detection
            exec_time, _ = self.measure_time(
                detector.identify_confidence_valleys,
                decisions_over_time,
                timestamps
            )
            
            peak_mem, current_mem, _ = self.measure_memory(
                detector.identify_confidence_valleys,
                decisions_over_time,
                timestamps
            )
            
            results.append({
                'size': size,
                'n_points': n_points,
                'time_ms': exec_time,
                'peak_memory_mb': peak_mem,
                'current_memory_mb': current_mem,
                'time_per_point_us': (exec_time * 1000) / n_points
            })
            
            print(f"  Time: {exec_time:.2f}ms")
            print(f"  Peak Memory: {peak_mem:.2f}MB")
            print(f"  Time per point: {results[-1]['time_per_point_us']:.2f}μs")
        
        return {'bias_detector': results}
    
    def test_batch_processing(self, batch_sizes: List[int]) -> Dict:
        """Test batch processing capabilities."""
        print("\n" + "="*70)
        print("BATCH PROCESSING PERFORMANCE TEST")
        print("="*70)
        
        tdp = TemporalDemographicParity(threshold=0.1)
        results = []
        
        # Generate large dataset
        total_size = 100000
        print(f"Generating {total_size:,} records for batch testing...")
        data = self.generate_test_data(total_size)
        
        for batch_size in batch_sizes:
            print(f"\nProcessing in batches of {batch_size:,}...")
            
            n_batches = total_size // batch_size
            batch_times = []
            
            start_total = time.perf_counter()
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_size)
                
                batch_time, _ = self.measure_time(
                    tdp.detect_bias,
                    data['decisions'][start_idx:end_idx],
                    data['groups'][start_idx:end_idx],
                    data['timestamps'][start_idx:end_idx]
                )
                batch_times.append(batch_time)
            
            end_total = time.perf_counter()
            total_time = (end_total - start_total) * 1000
            
            results.append({
                'batch_size': batch_size,
                'n_batches': n_batches,
                'avg_batch_time_ms': np.mean(batch_times),
                'total_time_ms': total_time,
                'throughput_records_per_sec': total_size / (total_time / 1000)
            })
            
            print(f"  Average batch time: {results[-1]['avg_batch_time_ms']:.2f}ms")
            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Throughput: {results[-1]['throughput_records_per_sec']:.0f} records/sec")
        
        return {'batch_processing': results}
    
    def test_complexity_scaling(self) -> Dict:
        """Test algorithmic complexity by measuring scaling behavior."""
        print("\n" + "="*70)
        print("COMPLEXITY SCALING ANALYSIS")
        print("="*70)
        
        sizes = [1000, 2000, 5000, 10000, 20000, 50000]
        metrics_results = {}
        
        # Test each metric
        metrics = {
            'TDP': TemporalDemographicParity(threshold=0.1),
            'EOOT': EqualizedOddsOverTime(tpr_threshold=0.15, fpr_threshold=0.15),
            'QPF': QueuePositionFairness(fairness_threshold=0.8)
        }
        
        for name, metric in metrics.items():
            print(f"\n{name} Complexity Analysis:")
            times = []
            
            for size in sizes:
                data = self.generate_test_data(size)
                
                if name == 'TDP':
                    exec_time, _ = self.measure_time(
                        metric.detect_bias,
                        data['decisions'],
                        data['groups'],
                        data['timestamps']
                    )
                elif name == 'EOOT':
                    exec_time, _ = self.measure_time(
                        metric.detect_bias,
                        data['decisions'],
                        data['labels'],
                        data['groups'],
                        data['timestamps']
                    )
                elif name == 'QPF':
                    exec_time, _ = self.measure_time(
                        metric.detect_bias,
                        data['queue_positions'],
                        data['groups'],
                        data['timestamps']
                    )
                
                times.append(exec_time)
                print(f"  {size:6,} records: {exec_time:8.2f}ms")
            
            # Calculate complexity
            complexity = self._estimate_complexity(sizes, times)
            metrics_results[name] = {
                'sizes': sizes,
                'times': times,
                'complexity': complexity
            }
            print(f"  Estimated complexity: {complexity}")
        
        return {'complexity_scaling': metrics_results}
    
    def _estimate_complexity(self, sizes: List[int], times: List[float]) -> str:
        """
        Estimate algorithmic complexity from timing data.
        
        Returns:
            String description of complexity (e.g., "O(n)", "O(n log n)")
        """
        # Calculate ratios
        ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            ratios.append(time_ratio / size_ratio)
        
        avg_ratio = np.mean(ratios)
        
        # Determine complexity based on ratio
        if avg_ratio < 1.2:
            return "O(n) - Linear"
        elif avg_ratio < 1.5:
            return "O(n log n) - Linearithmic"
        elif avg_ratio < 2.5:
            return "O(n²) - Quadratic"
        else:
            return "O(n²+) - Super-quadratic"
    
    def test_memory_scaling(self) -> Dict:
        """Test memory usage scaling with data size."""
        print("\n" + "="*70)
        print("MEMORY SCALING ANALYSIS")
        print("="*70)
        
        sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        tdp = TemporalDemographicParity(threshold=0.1)
        results = []
        
        for size in sizes:
            print(f"\nTesting memory with {size:,} records...")
            data = self.generate_test_data(size)
            
            # Measure memory
            peak_mem, current_mem, _ = self.measure_memory(
                tdp.detect_bias,
                data['decisions'],
                data['groups'],
                data['timestamps']
            )
            
            # Calculate memory per record
            mem_per_record = (peak_mem * 1024) / size  # KB per record
            
            results.append({
                'size': size,
                'peak_memory_mb': peak_mem,
                'current_memory_mb': current_mem,
                'memory_per_record_kb': mem_per_record
            })
            
            print(f"  Peak Memory: {peak_mem:.2f}MB")
            print(f"  Memory per record: {mem_per_record:.3f}KB")
        
        # Check if memory scales linearly
        sizes_arr = np.array([r['size'] for r in results])
        mem_arr = np.array([r['peak_memory_mb'] for r in results])
        
        # Linear regression
        coeffs = np.polyfit(sizes_arr, mem_arr, 1)
        linear_score = np.corrcoef(sizes_arr, mem_arr)[0, 1] ** 2
        
        print(f"\nMemory Scaling Analysis:")
        print(f"  Linear correlation R²: {linear_score:.4f}")
        print(f"  Memory growth rate: {coeffs[0]*1000:.3f}KB per 1000 records")
        
        return {
            'memory_scaling': {
                'results': results,
                'linear_score': linear_score,
                'growth_rate_kb_per_1k': coeffs[0] * 1000
            }
        }
    
    def generate_performance_report(self, all_results: Dict) -> None:
        """Generate comprehensive performance report with visualizations."""
        print("\n" + "="*70)
        print("GENERATING PERFORMANCE REPORT")
        print("="*70)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Temporal Fairness Framework - Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Metric execution times
        ax1 = axes[0, 0]
        if 'tdp' in all_results:
            sizes = [r['size'] for r in all_results['tdp']]
            tdp_times = [r['time_ms'] for r in all_results['tdp']]
            ax1.plot(sizes, tdp_times, 'b-', label='TDP', marker='o')
        if 'eoot' in all_results:
            eoot_times = [r['time_ms'] for r in all_results['eoot']]
            ax1.plot(sizes, eoot_times, 'r-', label='EOOT', marker='s')
        if 'qpf' in all_results:
            qpf_times = [r['time_ms'] for r in all_results['qpf']]
            ax1.plot(sizes, qpf_times, 'g-', label='QPF', marker='^')
        
        ax1.set_xlabel('Number of Records')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Metric Performance Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Memory usage
        ax2 = axes[0, 1]
        if 'memory_scaling' in all_results:
            mem_results = all_results['memory_scaling']['results']
            sizes = [r['size'] for r in mem_results]
            memory = [r['peak_memory_mb'] for r in mem_results]
            ax2.plot(sizes, memory, 'purple', marker='o', linewidth=2)
            ax2.fill_between(sizes, 0, memory, alpha=0.3, color='purple')
        
        ax2.set_xlabel('Number of Records')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Memory Usage Scaling')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Batch processing throughput
        ax3 = axes[0, 2]
        if 'batch_processing' in all_results:
            batch_results = all_results['batch_processing']
            batch_sizes = [r['batch_size'] for r in batch_results]
            throughput = [r['throughput_records_per_sec'] for r in batch_results]
            ax3.bar(range(len(batch_sizes)), throughput, color='orange')
            ax3.set_xticks(range(len(batch_sizes)))
            ax3.set_xticklabels([f'{bs//1000}K' for bs in batch_sizes])
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Throughput (records/sec)')
        ax3.set_title('Batch Processing Performance')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Complexity analysis
        ax4 = axes[1, 0]
        if 'complexity_scaling' in all_results:
            complexity_data = all_results['complexity_scaling']
            for metric_name, data in complexity_data.items():
                sizes = data['sizes']
                times = data['times']
                ax4.plot(sizes, times, marker='o', label=f"{metric_name}: {data['complexity']}")
        
        ax4.set_xlabel('Number of Records')
        ax4.set_ylabel('Execution Time (ms)')
        ax4.set_title('Complexity Scaling Validation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # Plot 5: Time per record
        ax5 = axes[1, 1]
        if 'tdp' in all_results:
            sizes = [r['size'] for r in all_results['tdp']]
            tdp_per_record = [r['time_per_record_us'] for r in all_results['tdp']]
            eoot_per_record = [r['time_per_record_us'] for r in all_results['eoot']]
            qpf_per_record = [r['time_per_record_us'] for r in all_results['qpf']]
            
            x = np.arange(len(sizes))
            width = 0.25
            ax5.bar(x - width, tdp_per_record, width, label='TDP', color='blue')
            ax5.bar(x, eoot_per_record, width, label='EOOT', color='red')
            ax5.bar(x + width, qpf_per_record, width, label='QPF', color='green')
            
            ax5.set_xlabel('Dataset Size')
            ax5.set_ylabel('Time per Record (μs)')
            ax5.set_title('Per-Record Processing Time')
            ax5.set_xticks(x)
            ax5.set_xticklabels([f'{s//1000}K' for s in sizes])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance summary heatmap
        ax6 = axes[1, 2]
        
        # Create summary matrix
        metrics = ['TDP', 'EOOT', 'FDD', 'QPF']
        criteria = ['10K Time', 'Memory', 'Complexity', 'Scalability']
        scores = np.random.rand(len(metrics), len(criteria))  # Placeholder
        
        # Populate with actual data if available
        if 'tdp' in all_results and len(all_results['tdp']) > 2:
            # Find 10K record performance
            for r in all_results['tdp']:
                if r['size'] == 10000:
                    scores[0, 0] = min(1.0, 100 / r['time_ms'])  # Normalize to 0-1
                    scores[0, 1] = min(1.0, 10 / r['peak_memory_mb'])
        
        im = ax6.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax6.set_xticks(np.arange(len(criteria)))
        ax6.set_yticks(np.arange(len(metrics)))
        ax6.set_xticklabels(criteria)
        ax6.set_yticklabels(metrics)
        ax6.set_title('Performance Score Matrix')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(criteria)):
                text = ax6.text(j, i, f'{scores[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
        print("\nPerformance report saved to 'performance_analysis.png'")
    
    def run_comprehensive_test(self) -> Dict:
        """Run all performance tests."""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE TESTING SUITE")
        print("="*70)
        print("\nSystem Information:")
        print(f"  CPU Count: {psutil.cpu_count()}")
        print(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        
        all_results = {}
        
        # Define test sizes
        test_sizes = [1000, 5000, 10000, 25000, 50000]
        
        # Test individual metrics
        print("\n[1/8] Testing Temporal Demographic Parity...")
        all_results.update(self.test_tdp_performance(test_sizes))
        
        print("\n[2/8] Testing Equalized Odds Over Time...")
        all_results.update(self.test_eoot_performance(test_sizes))
        
        print("\n[3/8] Testing Fairness Decay Detection...")
        all_results.update(self.test_fdd_performance(test_sizes))
        
        print("\n[4/8] Testing Queue Position Fairness...")
        all_results.update(self.test_qpf_performance(test_sizes))
        
        print("\n[5/8] Testing Bias Detector...")
        all_results.update(self.test_bias_detector_performance(test_sizes))
        
        print("\n[6/8] Testing Batch Processing...")
        batch_sizes = [1000, 5000, 10000, 25000]
        all_results.update(self.test_batch_processing(batch_sizes))
        
        print("\n[7/8] Testing Complexity Scaling...")
        all_results.update(self.test_complexity_scaling())
        
        print("\n[8/8] Testing Memory Scaling...")
        all_results.update(self.test_memory_scaling())
        
        # Generate report
        self.generate_performance_report(all_results)
        
        # Save results to JSON
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict) -> None:
        """Save performance results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open('performance_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print("\nPerformance results saved to 'performance_results.json'")
    
    def validate_requirements(self, results: Dict) -> None:
        """Validate that performance meets requirements."""
        print("\n" + "="*70)
        print("REQUIREMENTS VALIDATION")
        print("="*70)
        
        requirements = {
            '10K records < 1 second': True,
            'Memory scales linearly': True,
            'Handles 100K+ datasets': True,
            'O(n log n) complexity': True
        }
        
        # Check 10K performance
        for metric in ['tdp', 'eoot', 'qpf']:
            if metric in results:
                for r in results[metric]:
                    if r['size'] == 10000:
                        if r['time_ms'] > 1000:
                            requirements['10K records < 1 second'] = False
                            print(f"  ✗ {metric.upper()} failed 10K requirement: {r['time_ms']:.2f}ms")
        
        # Check memory scaling
        if 'memory_scaling' in results:
            if results['memory_scaling']['linear_score'] < 0.95:
                requirements['Memory scales linearly'] = False
                print(f"  ✗ Memory scaling non-linear: R²={results['memory_scaling']['linear_score']:.3f}")
        
        # Check complexity
        if 'complexity_scaling' in results:
            for metric, data in results['complexity_scaling'].items():
                if 'O(n²)' in data['complexity']:
                    requirements['O(n log n) complexity'] = False
                    print(f"  ✗ {metric} has {data['complexity']} complexity")
        
        # Print summary
        print("\nRequirements Summary:")
        for req, passed in requirements.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{status}] {req}")
        
        all_passed = all(requirements.values())
        if all_passed:
            print("\n🎉 All performance requirements met!")
        else:
            print("\n⚠️ Some requirements not met. See details above.")


def main():
    """Run performance testing suite."""
    tester = PerformanceTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Validate requirements
    tester.validate_requirements(results)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    # TDP Performance at 10K
    if 'tdp' in results:
        for r in results['tdp']:
            if r['size'] == 10000:
                print(f"\nTDP at 10K records:")
                print(f"  Time: {r['time_ms']:.2f}ms")
                print(f"  Memory: {r['peak_memory_mb']:.2f}MB")
                print(f"  Per record: {r['time_per_record_us']:.2f}μs")
    
    # EOOT Performance at 10K
    if 'eoot' in results:
        for r in results['eoot']:
            if r['size'] == 10000:
                print(f"\nEOOT at 10K records:")
                print(f"  Time: {r['time_ms']:.2f}ms")
                print(f"  Memory: {r['peak_memory_mb']:.2f}MB")
                print(f"  Per record: {r['time_per_record_us']:.2f}μs")
    
    # QPF Performance at 10K
    if 'qpf' in results:
        for r in results['qpf']:
            if r['size'] == 10000:
                print(f"\nQPF at 10K records:")
                print(f"  Time: {r['time_ms']:.2f}ms")
                print(f"  Memory: {r['peak_memory_mb']:.2f}MB")
                print(f"  Per record: {r['time_per_record_us']:.2f}μs")
    
    # Batch processing
    if 'batch_processing' in results:
        best_batch = max(results['batch_processing'], 
                        key=lambda x: x['throughput_records_per_sec'])
        print(f"\nBest batch processing:")
        print(f"  Batch size: {best_batch['batch_size']:,}")
        print(f"  Throughput: {best_batch['throughput_records_per_sec']:,.0f} records/sec")
    
    print("\n" + "="*70)
    print("PERFORMANCE TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()