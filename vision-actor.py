"""
This module extracts instructions using a specified model from OpenAI.
"""

import os
import argparse
from glob import glob
import base64
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
        # default="http://nlp-in-477-l:8001/v1",
        default="http://ucsc-real.soe.ucsc.edu:8000/v1",
        help="Model Server URL.",
    )
    # parser.add_argument('--model', type=str, default="McGill-NLP/Llama-3-8B-Web", help='Model to use for instruction extraction.')
    parser.add_argument(
        "--model",
        type=str,
        # default="llava-hf/llava-v1.6-mistral-7b-hf",
        # default="Qwen/Qwen2-VL-7B-Instruct",
        # default="google/paligemma2-3b-ft-docci-448",
        # default="llava-hf/llava-1.5-7b-hf",
        # default="deepseek-ai/deepseek-vl2",
        # default="Qwen/Qwen2.5-VL-7B-Instruct",
        # default="gpt-4o-mini",
        # default="Qwen/Qwen2.5-7B-Instruct-1M",
        # default="google/paligemma2-10b-pt-896",
        # default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        # default="microsoft/Phi-3.5-vision-instruct",
        # default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        default="openbmb/MiniCPM-o-2_6",
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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
    # Path to your image
    # image_path = "MCTSNode_SIMULATED_15_screenshot_som.png"

    for image_path in glob(
        "2025-02-14_19-36-19_GenericActingAgentArgs_on_webarena.4_7/simulator/MCTSNode_SIMULATED_0_screenshot_som.png"
    ):
        # Getting the Base64 string
        base64_image = encode_image(image_path)

        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Given a screenshot of a website, determine the status of the task: 'Navigate to the Best Sellers section.' Analyze the screenshot to assess progress and output two fields: choice and reason, separated by a comma. Keep the reason within 100 words. Select one of the following options for choice:**
completed, The subtask has been successfully completed.
close_to_completed, Only one simple action is needed to complete the subtask.
in_progress, The subtask is ongoing and requires more than one action to complete.
few_progress, Minimal or almost no progress has been made.
impossible, The subtask cannot be completed due to website constraints.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        options = [
            "completed",
            "close_to_completed",
            "in_progress",
            "few_progress",
            "impossible",
        ]
        start = time()
        choices = {}
        choice_text = {}
        for _ in range(1):
            r = client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=temperature,
                n=5,
                logprobs=True,
                top_logprobs=5,
            )
            for i, choice in enumerate(r.choices):
                for index, option in enumerate(options):
                    if option in choice.message.content:
                        if index in choices:
                            choices[index] += 1
                            choice_text[index] = choice.message.content
                        else:
                            choices[index] = 1
                            choice_text[index] = choice.message.content
        print(f"Time taken: {time() - start:.2f}s")
        # Find the option with the highest count
        if choices:
            highest_index = max(choices, key=choices.get)
            highest_option = options[highest_index]
            highest_count = choices[highest_index]
            print(
                f"For {image_path}, highest option: `{highest_option}` with count: {highest_count} and the text is: `{choice_text[highest_index]}`"
            )
        else:
            print(f"For {image_path}, No options found.")
