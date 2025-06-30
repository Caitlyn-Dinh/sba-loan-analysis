# This cell demonstrates how to access flow results
# Note: This will only work after you've actually run a flow

from metaflow import Flow

# Function to safely demonstrate flow access
def demonstrate_flow_access():
    try:
        # Try to access a flow (this will fail if no flows have been run)
        flow = Flow('WorkshopIntroFlow')
        latest_run = flow.latest_run
        # Check if run was successful
        if latest_run.successful:
            print('Run completed successfully!')

        # Access artifacts from specific steps
        end_step = latest_run['end']
        statistics = end_step.task.data.statistics

        # Access all artifacts
        for artifact_name in end_step.task.data:
            print(f'Artifact: {artifact_name}')
        
        print("📋 How to access flow results:")
        print("")
        print("# Get a specific flow")
        print("flow = Flow('WorkshopIntroFlow')")
        print("")
        print("# Get the latest run")
        print("latest_run = flow.latest_run")
        print("")
        print("# Check if run was successful")
        print("if latest_run.successful:")
        print("    print('Run completed successfully!')")
        print("")
        print("# Access artifacts from specific steps")
        print("end_step = latest_run['end']")
        print("statistics = end_step.task.data.statistics")
        print("")
        print("# Access all artifacts")
        print("for artifact_name in end_step.task.data:")
        print("    print(f'Artifact: {artifact_name}')")
        
        print("\n✅ Flow access patterns demonstrated!")
        print("💡 Run a flow first, then use these patterns to access results")
        
    except Exception as e:
        print(f"ℹ️  No flows run yet: {e}")
        print("This is expected - run a flow first to see real results!")

demonstrate_flow_access()

if __name__ == '__main__':
    demonstrate_flow_access()