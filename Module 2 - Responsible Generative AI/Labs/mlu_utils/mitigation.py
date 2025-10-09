import re
import random
import numpy as np
from collections import Counter
import json
import logging
import boto3
import time


from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate

from IPython.display import Markdown, display


def generate_text_mistral(model_id, body, temperature=0.0):
    """
    Generate text using a Mistral AI model.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        JSON: The response from the model.
    """

    # Each model has a different set of inference parameters
    inference_modifier = {
        "max_tokens": 2048,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 5
    }

    bedrock_runtime = boto3.client(service_name='bedrock-runtime')
    accept = "application/json"
    content_type = "application/json"
    
    # Define the langchain module with the selected bedrock model
    bedrock_llm = BedrockLLM(
        model_id=model_id, client=bedrock_runtime, model_kwargs=inference_modifier
    )

    try:
        response = bedrock_llm.invoke(body)
    except Exception as e:
        print(str(e))

    return response


def invoke_mistral(prompt:str, model:str='mistral.mistral-7b-instruct-v0:2', temperature:float=0.0, max_tokens:int=1000, stop_sequences:list=[], n:int=1):
    
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 5
    })

    response = generate_text_mistral(model_id=model, body=body, temperature=temperature)

    return response #outputs[0]['text']


def generate_text_titan(model_id, body, temperature=0.0):
    """
    Generate text using Amazon Titan Text models on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        response (json): The response from the model.
    """

    inference_modifier = {
        "maxTokenCount": 2048,
        "temperature": temperature,
        "topP": 0.9,
    }
    bedrock_runtime = boto3.client(service_name='bedrock-runtime')

    accept = "application/json"
    content_type = "application/json"

    bedrock_llm = BedrockLLM(
        model_id=model_id, client=bedrock_runtime, model_kwargs=inference_modifier
    )
    response = bedrock_llm.invoke(body)
    
    return response


def invoke_titan(prompt:str, model:str="amazon.titan-text-lite-v1", temperature:float=0.0, max_tokens:int=1000, stop_sequences:list=[], n:int=1):
    
    body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "stopSequences": stop_sequences,
                "temperature": temperature,
                "topP": 0.9
            }
        })

    response_body = generate_text_titan(model, body, temperature)
    
    return response_body 


def run_inference(prompt:str, model:str, temperature:float, max_tokens:int=1000, stop_sequences:list=[], n:int=1) -> list:
    """
    Function to run inference with models hosted in Bedrock
    """
    model_provider = model.split(".")[0]
    
    outputs = []
    for i in range(n):
        if model_provider == "mistral":
            outputs.append(invoke_mistral(prompt, model, temperature, max_tokens, stop_sequences, n))
        else:
            outputs.append(invoke_titan(prompt, model, temperature, max_tokens, stop_sequences, n))

    return outputs


def generate_outputs(prompt:str, model:str, temperature:float, max_tokens:int=1000, stop_sequences:list=[], n:int=1) -> list:
    """
    Function to wrap calls to inference functions. 
    """
    outputs = run_inference(prompt, model, temperature, max_tokens, stop_sequences, n)

    return outputs