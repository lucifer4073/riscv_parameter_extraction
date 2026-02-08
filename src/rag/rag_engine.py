from typing import List, Dict, Any
from pathlib import Path

class RAGEngine:
    def __init__(self, vector_store, llm_processor):
        self.vector_store = vector_store
        self.llm_processor = llm_processor
        
    def process_snippet(self, snippet_text: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved_schemas = self.vector_store.retrieve_top_k(snippet_text, k=top_k)
        
        primary_category = self._detect_primary_category(retrieved_schemas)
        
        system_prompt = self._build_system_prompt(retrieved_schemas, primary_category)
        user_prompt = self._build_user_prompt(snippet_text)
        
        response = self.llm_processor.generate(system_prompt, user_prompt)
        
        return {
            'response': response,
            'primary_category': primary_category,
            'retrieved_schemas': [s['schema_name'] for s in retrieved_schemas],
            'top_scores': [s['score'] for s in retrieved_schemas]
        }
    
    def _detect_primary_category(self, retrieved_schemas: List[Dict]) -> str:
        if not retrieved_schemas:
            return 'unknown'
        return retrieved_schemas[0]['schema_name']
    
    def _build_system_prompt(self, schemas: List[Dict], primary_category: str) -> str:
        prompt = f"""You are an expert RISC-V architectural parameter extraction system.

PRIMARY CATEGORY: {primary_category}

You are a Senior RISC-V ISA Verification Engineer and Architect. Your task is to analyze text snippets from the RISC-V Instruction Set Architecture (ISA) Manual and extract "Implementation Parameters."

### Definition of a Parameter
An "Implementation Parameter" is any aspect of the hardware or software environment that is NOT fixed by the ISA but is left to the discretion of the implementer. 

### Detection Triggers
Look for these specific linguistic markers:
1. **Explicit Flexibility:** "Implementation defined", "Implementation specific", "Not specified", "Profile-defined".
2. **Modal Verbs:** "May", "Might", "Should", "Optional", "Optionally", "Can be".
3. **Conditional Existence:** "If implemented", "If provided".
4. **Ranges/Limits:** "Up to", "Maximum of", "Minimum of", "at least".
5. **WARL/WLRL Fields:** If a register field is described as WARL (Write Any Read Legal), you must extract parameters for:
   - The set of legal values supported.
   - The mapping strategy for illegal values (how they are converted to legal ones).

RETRIEVED SCHEMA DEFINITIONS:
"""
        for i, schema in enumerate(schemas, 1):
            prompt += f"\n{i}. Schema: {schema['schema_name']}\n"
            prompt += f"   Content:\n{schema['content']}\n"
        
        prompt += """
### Output Rules
1. **Strict Grounding:** You must ignore prior knowledge of the UDB (Unified Database). You may ONLY extract parameters explicitly justified by the provided text snippets.
2. **Quote Source:** For every parameter, you must provide the exact substring from the text that implies its existence.
3. **Format:** Output purely in YAML format.

### YAML Schema
- name: [Short, descriptive variable name, snake_case]
  description: [What does this parameter control?]
  type: [Boolean, Integer, Enum, String, or Address Range]
  . . . etc.
Given the above parameters are just for reference ,and you are expected to find a larger number of relevant parameters. Your performance will be judged on the number and quality of parameters you can extract.

For example:
LLM- A extracts the following key parameters (name, description, type).
LLM- B extracts the following key parameters (name, kind, description, properties, address) .
LLM- B wil have a greater score than LLM - A.
"""
        
        return prompt
    
    def _build_user_prompt(self, snippet_text: str) -> str:
        return f"""RISC-V Specification Snippet:

{snippet_text}

Extract all architectural parameters in YAML format as specified."""