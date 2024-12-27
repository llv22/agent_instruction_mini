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


class TaskFromPlanner(BaseModel):
    id: int
    description: str


class AgentPlannerOutput(BaseModel):
    # taskId: int
    thought: Optional[str]
    keywords: Optional[List[str]]
    subtasks: Optional[List[TaskFromPlanner]]


def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default="http://nlp-in-477-l:8001/v1",
        help="Model Server URL.",
    )
    # parser.add_argument('--model', type=str, default="McGill-NLP/Llama-3-8B-Web", help='Model to use for instruction extraction.')
    parser.add_argument(
        "--model",
        type=str,
        # default="llava-hf/llava-v1.6-mistral-7b-hf",
        # default="Qwen/Qwen2-VL-7B-Instruct",
        default="google/paligemma2-3b-ft-docci-448",
        # default="llava-hf/llava-1.5-7b-hf",
        help="Model to use for instruction extraction.",
    )
    # parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='Model to use for instruction extraction.')
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = conf()
    openai_api_base, model, temperature = (
        args.openai_api_base,
        args.model,
        args.temperature,
    )
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = os.environ.get(f"{model}:api_key")
    openai_api_base = os.environ.get(f"{model}:url_base")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    payload = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
            },
        ],
        }
    ]

    start = time()
    r = client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=temperature,
    )
    print(
        f"\nTime taken: {time() - start:.2f}s, response: {r.choices[0].message.content}"
    )
