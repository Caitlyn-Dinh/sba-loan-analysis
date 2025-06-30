# Test Metaflow basic functionality
from metaflow import FlowSpec, step

class VerificationFlow(FlowSpec):
    """
    Simple flow to test Metaflow setup
    """
    
    @step
    def start(self):
        print("🚀 Metaflow verification starting...")
        self.message = "Hello from Metaflow!"
        self.next(self.end)
    
    @step
    def end(self):
        print(f"✅ {self.message}")
        print("🎉 Metaflow verification complete!")

# Test flow creation
try:
    flow = VerificationFlow()
    print("✅ Metaflow FlowSpec created successfully!")
except Exception as e:
    print(f"❌ Metaflow error: {e}")