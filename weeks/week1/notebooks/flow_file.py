# Import Metaflow components
from metaflow import FlowSpec, step
import pandas as pd
import numpy as np

class WorkshopIntroFlow(FlowSpec):
    """
    Our first Metaflow workflow - demonstrates core concepts
    """
    
    @step
    def start(self):
        """
        Initialize our workflow with sample data
        """
        print("🚀 Starting our first Metaflow workflow!")
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = {
            'values': np.random.normal(100, 15, 1000),
            'categories': np.random.choice(['A', 'B', 'C'], 1000),
            'timestamps': pd.date_range('2024-01-01', periods=1000)
        }
        
        print(f"✅ Generated {len(self.sample_data['values'])} data points")
        self.next(self.process_data)
    
    @step  
    def process_data(self):
        """
        Process our data and calculate statistics
        """
        print("🔧 Processing data...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.sample_data)
        
        # Calculate statistics
        self.statistics = {
            'mean': df['values'].mean(),
            'std': df['values'].std(),
            'count_by_category': df['categories'].value_counts().to_dict()
        }
        
        print(f"📊 Statistics calculated:")
        print(f"   Mean: {self.statistics['mean']:.2f}")
        print(f"   Std: {self.statistics['std']:.2f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Finalize workflow
        """
        print("🎉 Workflow completed successfully!")
        print(f"📋 Final statistics: {self.statistics}")

# Note: We define the flow here, but will run it from command line
print("✅ WorkshopIntroFlow defined successfully!")
print("💡 To run this flow, save it as a .py file and use: python flow_file.py run")

if __name__ == '__main__':
    WorkshopIntroFlow()