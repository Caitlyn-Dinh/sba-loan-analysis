from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import subprocess

print("💻 Exercise 2: Model Comparison Chain")
print("=" * 35)

# 🔍 Check if Ollama model is installed
def get_available_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return "llama3.2" in result.stdout
    except Exception as e:
        print("⚠️ Error checking models:", e)
        return False

available_models = get_available_models()

# 📊 Define the prompt template for model comparison
comparison_prompt = ChatPromptTemplate.from_template("""
You are a machine learning expert.

🔍 Problem Description: {problem}
📊 Dataset Characteristics: {dataset}

Compare and recommend models based on the following criteria:
- Accuracy
- Interpretability
- Training time
- Overfitting risk

🧠 Include trade-offs and justify your recommendation.
""")

# 🧠 Create the full chain
class ModelComparisonChain:
    def __init__(self, model_name="llama3.2", max_retries=3):
        self.model = OllamaLLM(model=model_name)
        self.output_parser = StrOutputParser()
        self.max_retries = max_retries
        self.chain = comparison_prompt | self.model | self.output_parser

    def compare_with_retry(self, problem, dataset):
        for attempt in range(self.max_retries):
            try:
                if not problem or not dataset:
                    raise ValueError("Problem and dataset description required")

                result = self.chain.invoke({
                    "problem": problem,
                    "dataset": dataset
                })

                if len(result.strip()) < 20:
                    raise ValueError("Output too short")

                return {
                    "success": True,
                    "response": result,
                    "attempt": attempt + 1
                }

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"success": False, "error": str(e), "attempts": attempt + 1}
                time.sleep(2 ** attempt)

# ✅ Run the comparison if the model is available
if available_models:
    print("\n🔧 Build your model comparison chain here:")

    chain = ModelComparisonChain()

    test_cases = [
        {
            "problem": "Predicting housing prices",
            "dataset": "Medium-sized tabular dataset with 50 features, some missing values"
        },
        {
            "problem": "Image classification for medical scans",
            "dataset": "Large dataset (100k+ images), high-dimensional, labeled"
        },
        {
            "problem": "Customer churn prediction",
            "dataset": "Small dataset with categorical variables and imbalanced classes"
        }
    ]

    for case in test_cases:
        print("\n📂 Scenario:", case["problem"])
        result = chain.compare_with_retry(case["problem"], case["dataset"])
        if result["success"]:
            print("\n📘 Model Comparison Output:\n")
            print(result["response"])
        else:
            print("❌ Error:", result["error"])
else:
    print("\n⚠️  Install Ollama and download a model (e.g. llama3.2) to complete this exercise")

print("\n✅ Exercise 2 space ready!")
