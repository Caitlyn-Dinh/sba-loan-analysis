import subprocess
import sys
import os

def check_environment():
    """Comprehensive environment verification for Week 2 workshop"""
    print("🔍 WEEK 2 ENVIRONMENT CHECK")
    print("=" * 40)
    
    # Corrected: PyPI name → import name + version
    required_packages = {
        'metaflow': ('metaflow', '2.7+'),
        'pandas': ('pandas', '1.3+'),
        'numpy': ('numpy', '1.20+'),
        'scikit-learn': ('sklearn', '1.0+'),
        'langchain': ('langchain', '0.1+'),
        'langchain-community': ('langchain_community', '0.0.10+'),
        'matplotlib': ('matplotlib', '3.3+'),
        'seaborn': ('seaborn', '0.11+')
    }

    print("📦 Checking Python packages...")
    missing_packages = []

    for pip_name, (import_name, version) in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {pip_name} {version}")
        except ImportError:
            print(f"   ❌ {pip_name} {version} - MISSING")
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing_packages)}")
        return False

    # Check data files
    print("\n📊 Checking data files...")
    data_files = [
        '../data/titanic.csv',
        '../data/customer_reviews.csv',
        '../data/financial_data.json'
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"   ✅ {os.path.basename(file_path)}")
        else:
            print(f"   ⚠️  {os.path.basename(file_path)} - Not found (will use sample data)")

    # Check Ollama
    print("\n🧠 Checking Ollama installation...")
    try:
        result = subprocess.run(['ollama', '--version'], 
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ Ollama installed")
            print("   💡 Download model with: ollama pull llama3.2")
        else:
            print("   ❌ Ollama command failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ Ollama not found - install from ollama.com")
        print("      This is required for LLM exercises")

    print("\n🎯 Environment check complete!")
    print("   Ready to start the workshop!")
    return True

# Run the environment check
if __name__ == "__main__":
    check_environment()
