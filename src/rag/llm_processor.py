import ollama
from typing import Dict, Any

class LLMProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def check_availability(self) -> bool:
        try:
            ollama.list()
            return True
        except:
            return False