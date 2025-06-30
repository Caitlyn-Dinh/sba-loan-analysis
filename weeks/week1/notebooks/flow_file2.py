# Let's explore the flow structure in more detail
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np

class DetailedFlow(FlowSpec):
    """
    A more detailed flow showing Metaflow features
    """
    
    # Parameters allow customization when running
    sample_size = Parameter('sample_size', 
                           help='Number of samples to generate',
                           default=100)
    
    @step
    def start(self):
        """
        Generate data with configurable size
        """
        print(f"🎯 Generating {self.sample_size} samples")
        
        # Generate data using the parameter
        np.random.seed(42)
        self.data = np.random.normal(0, 1, self.sample_size)
        
        # Store metadata about our data
        self.metadata = {
            'created_at': pd.Timestamp.now(),
            'sample_size': len(self.data),
            'data_type': 'normal_distribution'
        }
        
        print(f"✅ Data generated: shape {self.data.shape}")
        self.next(self.analyze)
    
    @step
    def analyze(self):
        """
        Analyze the generated data
        """
        print("📊 Analyzing data...")
        
        # Calculate comprehensive statistics
        self.analysis = {
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'percentiles': {
                '25th': float(np.percentile(self.data, 25)),
                '50th': float(np.percentile(self.data, 50)),
                '75th': float(np.percentile(self.data, 75))
            }
        }
        
        print(f"   Mean: {self.analysis['mean']:.3f}")
        print(f"   Std: {self.analysis['std']:.3f}")
        print(f"   Range: [{self.analysis['min']:.3f}, {self.analysis['max']:.3f}]")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Summarize results
        """
        print("🎉 Analysis complete!")
        
        # Create final summary
        self.summary = {
            'workflow': 'DetailedFlow',
            'parameters': {'sample_size': self.sample_size},
            'metadata': self.metadata,
            'results': self.analysis
        }
        
        print(f"📋 Summary created with {len(self.summary)} sections")

print("✅ DetailedFlow defined successfully!")
print("💡 This flow demonstrates parameters and comprehensive data tracking")

if __name__ == '__main__':
    DetailedFlow()