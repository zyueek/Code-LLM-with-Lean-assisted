#!/usr/bin/env python3
"""
vLLM client for connecting to localhost:1234
Based on the existing usage in RAG/llm_code_generator.py
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional

class VLLMClient:
    """Client for connecting to an OpenAI-compatible vLLM server.

    Supports both /v1/completions and /v1/chat/completions. Automatically
    normalizes the base_url and falls back to chat if text completions return
    a 4xx error (common when servers expose only chat endpoints).
    """

    def __init__(self, base_url: str = "http://localhost:1234", model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"):
        # Normalize base URL to a rooted /v1, then derive both endpoints
        raw = (base_url or "http://localhost:1234").rstrip("/")
        if raw.endswith("/v1/chat/completions"):
            root_v1 = raw[: -len("/chat/completions")]
        elif raw.endswith("/v1/completions"):
            root_v1 = raw[: -len("/completions")]
        elif raw.endswith("/v1"):
            root_v1 = raw
        else:
            root_v1 = f"{raw}/v1"

        self.root_v1 = root_v1
        self.completions_url = f"{self.root_v1}/completions"
        self.chat_completions_url = f"{self.root_v1}/chat/completions"

        # Default primary endpoint (try text completions first)
        self.vllm_url = self.completions_url

        # Allow environment override for model without changing callers
        env_model = os.getenv("PROVER_MODEL")
        self.model_name = env_model if env_model else model_name
        print(f"Initialized vLLM client with model {self.model_name}")
        print(f"  - completions:     {self.completions_url}")
        print(f"  - chat.completions:{self.chat_completions_url}")
        
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, stop: Optional[List[str]] = None) -> str:
        """Generate text completion using vLLM - based on RAG/llm_code_generator.py pattern"""
        try:
            print(f"🔍 Sending prompt to LLM (first 200 chars): {prompt[:200]}...")

            # Attempt text completions first
            comp_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                comp_payload["temperature"] = temperature
            comp_payload["top_p"] = 0.9
            if stop:
                comp_payload["stop"] = stop
            response = requests.post(self.completions_url, json=comp_payload, headers={"Content-Type": "application/json"}, timeout=60)

            if response.status_code >= 400:
                try:
                    err_body = response.text
                    print(f"⚠️ completions error {response.status_code}: {err_body[:300]}")
                except Exception:
                    pass
                # Fallback to chat completions (many servers only expose chat)
                chat_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if temperature is not None:
                    chat_payload["temperature"] = temperature
                chat_payload["top_p"] = 0.9
                if stop:
                    chat_payload["stop"] = stop
                response_chat = requests.post(self.chat_completions_url, json=chat_payload, headers={"Content-Type": "application/json"}, timeout=60)
                if response_chat.status_code >= 400:
                    try:
                        err_body = response_chat.text
                        print(f"⚠️ chat.completions error {response_chat.status_code}: {err_body[:300]}")
                    except Exception:
                        pass
                response_chat.raise_for_status()
                data = response_chat.json()
                generated = (data.get("choices") or [{}])[0]
                msg = (generated.get("message") or {})
                text = (msg.get("content") or "").strip()
                print(f"✅ LLM (chat) generated {len(text)} characters")
                return text

            # If completions succeeded
            response.raise_for_status()
            data = response.json()
            text = (data.get("choices") or [{}])[0].get("text", "").strip()
            print(f"✅ LLM (completions) generated {len(text)} characters")
            return text
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to VLLM server: {e}")
            return ""
        except Exception as e:
            print(f"❌ Error generating code: {e}")
            return ""
    
    def test_connection(self) -> bool:
        """Test if the vLLM server is responding - based on RAG/llm_code_generator.py pattern"""
        try:
            # Check if server is accessible by trying to get models endpoint
            models_url = f"{self.root_v1}/models"
            response = requests.get(models_url, timeout=5)
            print("✅ VLLM server is accessible")
            try:
                payload = response.json()
                ids = [m.get("id") for m in (payload.get("data") or []) if isinstance(m, dict)]
                if ids:
                    print(f"📦 Available models: {', '.join(ids[:6])}{' ...' if len(ids) > 6 else ''}")
                    if self.model_name not in ids:
                        print(f"⚠️ Requested model not listed: {self.model_name}")
            except Exception:
                pass
            
            # Also test a simple generation
            test_prompt = "def hello():"
            response = self.generate(test_prompt, max_tokens=10)
            if response:
                print(f"✅ Connection test successful. Response: {response[:50]}...")
                return True
            else:
                print("⚠️ Server accessible but generation failed")
                return False
                
        except Exception as e:
            print(f"⚠️ VLLM server not accessible: {e}")
            print("Make sure VLLM server is running on localhost:1234")
            return False

def main():
    """Test the vLLM client"""
    client = VLLMClient()
    
    print("Testing vLLM client connection to localhost:1234...")
    if client.test_connection():
        print("✓ Connection successful!")
    else:
        print("✗ Connection failed!")
        print("Make sure vLLM server is running on localhost:1234")

if __name__ == "__main__":
    main()
