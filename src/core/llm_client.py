import os
import json
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# load environment variables
load_dotenv()

class LLMClient:
    """Unified client supporting Fireworks, OpenRouter, Groq, and OpenAI."""
    
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url if base_url.endswith("/") else f"{base_url}/"
        self.api_key = api_key
        self.model = model_name
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def structured_completion(self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        """Call LLM with enforced JSON schema."""
        # Using JSON object mode compatible across all open-source API providers
        schema_json = json.dumps(response_schema.model_json_schema())
        augmented_system = f"{system_prompt}\n\nYou MUST return a valid JSON object strictly matching this schema:\n{schema_json}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw_text = response.choices[0].message.content
        return response_schema.model_validate_json(raw_text)

    def text_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Standard text generation for Reader and Judge."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content or ""

#make a function that will allow you to test this function locally.
def test_above_client():

    api_key = (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        api_key = "fake-key"
    
    client = LLMClient(base_url="https://api.groq.com/openai/v1", api_key=api_key, model_name="openai/gpt-oss-20b")

    response = client.text_completion("You are a helpful assistant.", "Hello, how are you?")
    print(response)

if __name__ == "__main__":
    test_above_client()


    