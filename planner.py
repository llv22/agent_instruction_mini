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
        default="http://nlp-in-477-l:8000/v1",
        help="Model Server URL.",
    )
    # parser.add_argument('--model', type=str, default="McGill-NLP/Llama-3-8B-Web", help='Model to use for instruction extraction.')
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
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
    # see: 128k context of llama3.1
    # completion = client.completions.create(model=model, prompt="San Francisco is a")
    payload = [
        {
            "role": "user",
            "content": 'You are a web automation task planner. You will receive tasks from the user and will work with a naive AI Helper agent to accomplish it.\nYou will think step by step and break down the tasks into sequence of simple tasks. Tasks will be delegated to the Helper to execute on browser. \n\n1. Your job is to do planning for a web agent, which intends to finish the intent in certain web site.\n2. You are given the domain of website and the intent. Based on this information, you are expected to identify important keywords from the intent and identify steps to achieve the intent on the web site.\n\nYour input and output will strictly be a well-formatted JSON with attributes as mentioned below. \n\nInput:\n- sites: The application domain that the web site has been identified. One web site may have multiple domains concatenated by ",", e.g. "map, shopping"\n- intent: Mandatory string representing the main objective to be achieved via web automation\n\nOutput:\n- thought: Mandatory string specifying your thoughts of why did you come up with the plan. Illustrate your reasoning here.\n- keywords: Mandatory list of strings representing the keywords extracted from the intent. Use these keywords to come up with the plan.\n- subtasks: Mandatory List of tasks that need be performed to achieve the intent. Think step by step. Each step will contains integer id with description of the task.\n\nExample 1:\nInput: {\n    "sites": "shopping_admin",\n    "intent": "What is the top-1 best-selling product in 2022"\n}\nOutput:\n{\n    "thought": "I see the intent is to find the top-1 best-selling product in 2022. I should first go to the shopping_admin site and then look for the best-selling products. I should then sort the products by sales and extract the top-1 product from the list.",\n    "keywords": ["top-1", "best-selling product", "2022"],\n    "subtasks": [\n        {"id": 1, "description": "Navigate to the \'Best Sellers\' section of the website"},\n        {"id": 2, "description": "Filter or sort the products by year to select 2022"},\n        {"id": 3, "description": "Identify the product with the highest sales in the filtered results"},\n        {"id": 4, "description": "Review the product details to confirm it is the top-selling item"},\n    ]\n}\n\nGiven the following input \n\n{"sites":"shopping_admin","intent":"What are the top-3 best-selling product in Jan 2023"}\n, please just generate the corresponding output in the json format.',
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

    start = time()
    client = instructor.from_openai(client, 
                                    # mode=Mode.JSON
                                    ) # see: refer to https://github.com/instructor-ai/instructor/discussions/806
    response = client.chat.completions.create(
        model=model,
        messages=payload,
        response_model=AgentPlannerOutput,
        temperature=temperature,
        max_retries=1,
    )
    print(
        f"\nTime taken: {time() - start:.2f}s, response: {response.model_dump_json()}"
    )
