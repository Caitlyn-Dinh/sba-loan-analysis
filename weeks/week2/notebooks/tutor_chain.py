from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

print("\n🔧 Production-Ready Tutor Chain:")

class TutorChain:
    """Production-grade tutor chain for explaining data science concepts"""
    
    def __init__(self, model, max_retries=3, timeout=30):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.output_parser = StrOutputParser()
        self.setup_chain()
    
    def setup_chain(self):
        """Initialize prompt + chain"""
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
        """Execute tutor chain with input validation and retries"""
        for attempt in range(self.max_retries):
            try:
                # Input validation
                if not concept or level not in ['beginner', 'intermediate', 'advanced']:
                    raise ValueError("Invalid concept or level")

                # Run chain
                result = self.chain.invoke({"concept": concept, "level": level})
                
                # Output validation
                if len(result.strip()) < 20:
                    raise ValueError("Response too short to be useful")
                
                return {
                    "success": True,
                    "response": result,
                    "attempt": attempt + 1
                }
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff

print("   ✅ Reusable LangChain-style tutor chain")
print("   ✅ Includes error handling, retries, and validation")
print("   ✅ Supports multiple levels and concepts")
