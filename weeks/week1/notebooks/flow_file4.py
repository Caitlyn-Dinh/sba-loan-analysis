from metaflow import FlowSpec, step, Parameter, catch
import pandas as pd
import numpy as np
from datetime import datetime

class ProductionReadyFlow(FlowSpec):
    """
    A production-ready flow demonstrating best practices:
    - Comprehensive error handling
    - Detailed logging
    - Parameter validation
    - Artifact organization
    """
    
    # Well-documented parameters with validation
    data_size = Parameter('data_size',
                         help='Number of data points to generate (10-10000)',
                         default=1000,
                         type=int)
    
    noise_level = Parameter('noise_level',
                           help='Noise level for data generation (0.1-2.0)',
                           default=0.5,
                           type=float)
    
    @step
    def start(self):
        """
        Initialize workflow with parameter validation and logging
        """
        print("🚀 Starting ProductionReadyFlow")
        print("=" * 40)
        print(f"Parameters:")
        print(f"  data_size: {self.data_size}")
        print(f"  noise_level: {self.noise_level}")
        
        # Parameter validation
        if not (10 <= self.data_size <= 10000):
            raise ValueError(f"data_size must be between 10 and 10000, got {self.data_size}")
        
        if not (0.1 <= self.noise_level <= 2.0):
            raise ValueError(f"noise_level must be between 0.1 and 2.0, got {self.noise_level}")
        
        # Store run metadata
        self.run_metadata = {
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'data_size': self.data_size,
                'noise_level': self.noise_level
            },
            'version': '1.0.0',
            'description': 'Production-ready data processing workflow'
        }
        
        print("✅ Parameters validated and metadata stored")
        self.next(self.generate_data)
    
    @catch(var='generation_error')
    @step
    def generate_data(self):
        """
        Generate synthetic data with error handling
        """
        print("\n🔧 Generating synthetic data...")
        
        try:
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Generate base signal
            x = np.linspace(0, 4*np.pi, self.data_size)
            signal = np.sin(x) + 0.5*np.cos(2*x)
            
            # Add noise
            noise = np.random.normal(0, self.noise_level, self.data_size)
            self.data = signal + noise
            
            # Create additional features
            self.features = {
                'x': x,
                'signal': signal,
                'noise': noise,
                'final_data': self.data
            }
            
            # Store generation info
            self.generation_info = {
                'data_points_generated': len(self.data),
                'signal_range': [float(signal.min()), float(signal.max())],
                'noise_std': float(noise.std()),
                'final_data_range': [float(self.data.min()), float(self.data.max())]
            }
            
            print(f"   ✅ Generated {len(self.data)} data points")
            print(f"   📊 Data range: [{self.data.min():.3f}, {self.data.max():.3f}]")
            print(f"   🔊 Noise std: {noise.std():.3f}")
            
            self.generation_error = None  # No error occurred
            
        except Exception as e:
            print(f"   ❌ Data generation failed: {e}")
            self.generation_error = str(e)
            # Set default empty data
            self.data = np.array([])
            self.features = {}
            self.generation_info = {'error': str(e)}
        
        self.next(self.analyze_data)
    
    @step
    def analyze_data(self):
        """
        Comprehensive data analysis
        """
        print("\n📊 Analyzing data...")
        
        if self.generation_error:
            print(f"   ⚠️ Skipping analysis due to generation error: {self.generation_error}")
            self.analysis_results = {'error': 'No data to analyze'}
        else:
            # Comprehensive statistical analysis
            self.analysis_results = {
                'basic_stats': {
                    'mean': float(np.mean(self.data)),
                    'std': float(np.std(self.data)),
                    'median': float(np.median(self.data)),
                    'min': float(np.min(self.data)),
                    'max': float(np.max(self.data))
                },
                'distribution_stats': {
                    'skewness': float(self._calculate_skewness(self.data)),
                    'kurtosis': float(self._calculate_kurtosis(self.data))
                },
                'percentiles': {
                    '10th': float(np.percentile(self.data, 10)),
                    '25th': float(np.percentile(self.data, 25)),
                    '75th': float(np.percentile(self.data, 75)),
                    '90th': float(np.percentile(self.data, 90))
                }
            }
            
            print(f"   📈 Mean: {self.analysis_results['basic_stats']['mean']:.3f}")
            print(f"   📊 Std: {self.analysis_results['basic_stats']['std']:.3f}")
            print(f"   📏 Range: [{self.analysis_results['basic_stats']['min']:.3f}, {self.analysis_results['basic_stats']['max']:.3f}]")
        
        self.next(self.end)
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    @step
    def end(self):
        """
        Finalize workflow with comprehensive summary
        """
        print("\n🎉 ProductionReadyFlow completed!")
        print("=" * 40)
        
        # Create comprehensive final report
        self.final_report = {
            'workflow_info': {
                'name': 'ProductionReadyFlow',
                'version': self.run_metadata['version'],
                'completion_time': datetime.now().isoformat()
            },
            'execution_summary': {
                'parameters_used': self.run_metadata['parameters'],
                'data_generation_successful': self.generation_error is None,
                'analysis_completed': 'error' not in self.analysis_results
            },
            'results': self.analysis_results if hasattr(self, 'analysis_results') else {},
            'metadata': self.run_metadata
        }
        
        print("📋 Final Report Summary:")
        print(f"   ✅ Workflow: {self.final_report['workflow_info']['name']}")
        print(f"   📊 Data generation: {'✅ Success' if self.final_report['execution_summary']['data_generation_successful'] else '❌ Failed'}")
        print(f"   🔍 Analysis: {'✅ Complete' if self.final_report['execution_summary']['analysis_completed'] else '❌ Failed'}")
        
        if self.generation_error is None:
            print(f"   📈 Data points processed: {len(self.data)}")
            print(f"   🎯 Final data mean: {self.analysis_results['basic_stats']['mean']:.3f}")
        
        print("\n💾 All artifacts saved automatically by Metaflow!")

print("✅ ProductionReadyFlow defined successfully!")
print("🏆 This flow demonstrates production-ready patterns:")
print("   - Parameter validation")
print("   - Error handling with @catch")
print("   - Comprehensive logging")
print("   - Detailed artifact organization")
print("   - Professional reporting")

if __name__ == '__main__':
    ProductionReadyFlow()