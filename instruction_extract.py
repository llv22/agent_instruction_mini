"""
This module extracts instructions using a specified model from OpenAI.
"""
import os
import argparse
import instructor
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from time import time

from instructor import Mode
from typing import Callable, List, Optional, Tuple, Type

# see: load environment variables from .env file
load_dotenv()

class UserInfo(BaseModel):
    keywords: Optional[List[str]]
    steps: Optional[List[str]]

def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_base', type=str, default="http://nlp-in-477-l:8000/v1", help='Model Server URL.')
    # parser.add_argument('--model', type=str, default="McGill-NLP/Llama-3-8B-Web", help='Model to use for instruction extraction.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model to use for instruction extraction.')
    # parser.add_argument('--model', type=str, default="allenai/Llama-3.1-Tulu-3-8B-SFT", help='Model to use for instruction extraction.')
    # parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='Model to use for instruction extraction.')
    return parser.parse_args()

if __name__ == "__main__":
    args = conf()
    openai_api_base, model = args.openai_api_base, args.model
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = os.environ.get(f"{model}:api_key")
    openai_api_base = os.environ.get(f"{model}:url_base")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # see: 128k context of llama3.1
    # completion = client.completions.create(model=model, prompt="San Francisco is a")
    payload = [
        {
            "role": "user",
            "content": "You are an assistant to help users extract the most informative instructions from the input."
        },
        {
            "role": "user",
            "content":  "You are given a shipping admin site and want to finish the intent 'What are the top-3 best-selling products in Jan 2023'. Could you please just return the following fields in the json format 1. keywords: identify important keywords in list from the intent and 2. steps: identify steps in list to achieve the intent on the web site?"
        }
    ]
    
    start = time()
    
    # see: original type 1
    r = client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=1.0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        n=20,
        logprobs=True,
    )
    print(f"\nTime taken: {time() - start:.2f}s, response: {r.choices[0].message.content}")
    
    start = time()
    client = instructor.from_openai(client)
    # client = instructor.from_openai(client, mode=Mode.JSON)
    response = client.chat.completions.create(
        model=model,
        messages=payload,
        response_model=UserInfo,
        temperature=0.6,
        max_retries=1,
    )
    print(f"\nTime taken: {time() - start:.2f}s, response: {response.model_dump_json()}")
