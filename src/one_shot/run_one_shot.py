# Install required packages:
# pip install litellm python-dotenv pyyaml

import os
from litellm import completion
import litellm
from dotenv import load_dotenv
import yaml
from datetime import datetime

# Load API keys from .env file
load_dotenv()
litellm.drop_params = True
def test_agentic_framework(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
):
    """
    Universal function to test prompts across different LLM providers.
    
    Args:
        system_prompt: System instruction for the model
        user_prompt: User query/input
        model: Model identifier (see examples below)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        **kwargs: Additional model-specific parameters
    
    Returns:
        dict: Response with content and metadata
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "model": model
        }


# Example Usage
if __name__ == "__main__":
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)
    
    # Define your prompts
    try:
        with open("prompts/system_prompt_1.txt", 'r', encoding='utf-8') as file:
            system_prompt = file.read()
    except FileNotFoundError:
        print("⚠️  Warning: prompts/system_prompt_1.txt not found. Using default.")
        system_prompt = "You are a helpful AI assistant."
    
    try:
        with open("prompts/user_prompt_2.txt", 'r', encoding='utf-8') as file:
            user_prompt = file.read()
    except FileNotFoundError:
        print("⚠️  Warning: prompts/user_prompt_1.txt not found. Using default.")
        user_prompt = "Hello, how can you help me?"
    
    # Test with different models
    models_to_test = [
        "claude-4.5-sonnet",
        "gemini-3-flash",
        "gemini-3-pro",
        "gpt-5",
        "grok-4.1"
    ]
    
    print("Testing Agentic Framework Across Models\n" + "="*50)
    
    all_results = []  # Store all results
    
    for model in models_to_test:
        print(f"\n🤖 Testing: {model}")
        print("-" * 50)
        
        result = test_agentic_framework(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=0.7,
            max_tokens=500
        )
        
        # Add metadata to result
        result["tested_at"] = datetime.now().isoformat()
        result["system_prompt"] = system_prompt
        result["user_prompt"] = user_prompt
        
        # Generate filename safely
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handle model names with and without '/'
        if '/' in model:
            model_name = model.split('/')[-1]  # Get last part after '/'
        else:
            model_name = model
        
        # Sanitize filename (remove any invalid characters)
        model_name = model_name.replace('/', '_').replace(':', '_').replace(' ', '_')
        filename = f"outputs/{model_name}_test_results_{timestamp}.yaml"
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            # Save error to YAML too
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        else:
            print(f"✅ Response:\n{result['content'][:200]}...")  # Show first 200 chars
            print(f"\n📊 Tokens Used: {result['usage']['total_tokens']}")
            
            response_content = result['content']
            
            # Try to parse if the response itself is YAML
            try:
                # Attempt to load the response as YAML
                parsed_content = yaml.safe_load(response_content)
                
                # Save the parsed YAML structure
                with open(filename, 'w', encoding='utf-8') as f:
                    yaml.dump(parsed_content, f, 
                             default_flow_style=False, 
                             allow_unicode=True, 
                             indent=2,
                             sort_keys=False)
                
                print(f"💾 Saved parsed YAML to: {filename}")
                
            except yaml.YAMLError as e:
                # If it's not valid YAML, save as plain text in YAML format
                print(f"⚠️  Response is not valid YAML, saving as text")
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(response_content)
                
                print(f"💾 Saved as plain text to: {filename}")
        
        all_results.append(result)
    
    # # Save consolidated results
    # consolidated_filename = f"outputs/all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    # consolidated_data = {
    #     "test_metadata": {
    #         "timestamp": datetime.now().isoformat(),
    #         "total_models_tested": len(models_to_test),
    #         "models": models_to_test
    #     },
    #     "results": all_results
    # }
    
    # with open(consolidated_filename, 'w', encoding='utf-8') as f:
    #     yaml.dump(consolidated_data, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    # print(f"\n📦 Consolidated results saved to: {consolidated_filename}")
        # Save response content to a YAML file
        
        


# Advanced: Function calling / Tool use example
def test_with_tools(system_prompt: str, user_prompt: str, model: str):
    """
    Example of using tools/function calling (supported by Claude, GPT-4, Gemini)
    """
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    return response