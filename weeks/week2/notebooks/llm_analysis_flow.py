from metaflow import FlowSpec, step
import pandas as pd
import numpy as np

# LangChain with Ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class LLMDataAnalysisFlow(FlowSpec):

    @step
    def start(self):
        print("📥 Loading data...")
        np.random.seed(42)
        self.df = pd.DataFrame({
            "age": np.random.normal(35, 10, 100),
            "income": np.random.normal(70000, 15000, 100),
            "purchased": np.random.choice([0, 1], size=100)
        })
        print("✅ Data loaded.")
        self.next(self.statistical_analysis, self.llm_analysis)

    @step
    def statistical_analysis(self):
        print("📊 Performing traditional statistical analysis...")
        df = self.df
        self.stats_summary = {
            "mean_age": df["age"].mean(),
            "mean_income": df["income"].mean(),
            "purchase_rate": df["purchased"].mean()
        }
        print("✅ Stats:", self.stats_summary)
        self.next(self.join)

    @step
    def llm_analysis(self):
        print("🤖 Generating insights using Ollama...")

        df_sample = self.df.head(10).to_csv(index=False)

        prompt_template = PromptTemplate(
            input_variables=["data"],
            template="""
You are a data analyst. Here is a CSV-formatted sample of a dataset:

{data}

Provide insights or trends you notice. Be concise and focus on business-relevant insights.
"""
        )

        # 👇 Use Ollama (must have `ollama run llama3` running in background)
        ollama_llm = Ollama(model="llama3")

        chain = LLMChain(llm=ollama_llm, prompt=prompt_template)
        self.llm_output = chain.run({"data": df_sample})
        print("✅ LLM insights generated.")
        self.next(self.join)

    @step
    def join(self, inputs):
        print("🔗 Merging results...")
        self.stats_summary = inputs.statistical_analysis.stats_summary
        self.llm_output = inputs.llm_analysis.llm_output
        self.next(self.report)

    @step
    def report(self):
        print("\n📋 Final Report:")
        print("=== Traditional Statistical Analysis ===")
        for k, v in self.stats_summary.items():
            print(f"{k}: {v:.2f}")
        print("\n=== LLM-Generated Insights (Ollama) ===")
        print(self.llm_output)
        self.next(self.end)

    @step
    def end(self):
        print("🏁 Flow completed.")


if __name__ == "__main__":
    LLMDataAnalysisFlow()
