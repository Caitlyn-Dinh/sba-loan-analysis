from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

class TutorChain:
    def __init__(self, model, max_retries=3):
        self.model = model
        self.max_retries = max_retries
        self.output_parser = StrOutputParser()
        self.setup_chain()
    
    def setup_chain(self):
        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful Data Science tutor.

📘 Concept: {concept}
🎓 Audience Level: {level}

Please:
1. Explain the concept clearly for a {level} learner.
2. Provide a real-world example or practical use case.
3. Ask 2 follow-up questions to test understanding.

Keep your tone warm and supportive.
""")
        self.chain = self.prompt | self.model | self.output_parser
    
    def teach_with_retry(self, concept, level):
        for attempt in range(self.max_retries):
            try:
                if not concept or level not in ['beginner', 'intermediate', 'advanced']:
                    raise ValueError("Invalid concept or level")
                result = self.chain.invoke({"concept": concept, "level": level})
                if len(result.strip()) < 20:
                    raise ValueError("Response too short")
                return {"success": True, "response": result, "attempt": attempt + 1}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"success": False, "error": str(e), "attempts": attempt + 1}
                time.sleep(2 ** attempt)

# Use LLaMA 3.2
model = Ollama(model="llama3.2")
tutor = TutorChain(model)

# Run a test
result = tutor.teach_with_retry("Baking", "advanced")

if result["success"]:
    print("\n📘 Tutor Output:\n")
    print(result["response"])
else:
    print("\n❌ Error:", result["error"])
