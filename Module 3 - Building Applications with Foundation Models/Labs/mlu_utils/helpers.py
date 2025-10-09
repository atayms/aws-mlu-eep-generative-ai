import boto3
from botocore.exceptions import ClientError
import json

# Common inference parameters for Converse API
converse_inference_params = {
    "maxTokens": "(int) Maximum number of tokens to generate",
    "temperature": "(float) Temperature for sampling",
    "topP": "(float) Top P for nucleus sampling",
    "stopSequences": "([string]) Sequences that stop generation"
}

def validate_inference_parameters(inference_config):
    """Validates inference parameters for Converse API"""
    for key in inference_config:
        if key not in converse_inference_params:
            raise ValueError(f"'{key}' is not a valid inference parameter for Converse API")
    return True

def validate_model_access(model_id):
    """Validates access to requested model using Converse API.
    
    Return True when model is accessible and False otherwise
    """
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")
    
    try:
        # Simple test request using Converse API format
        request = {
            "modelId": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "How are you?"
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 64,
                "temperature": 0.5
            }
        }
        
        bedrock_runtime.converse(**request)
        return True
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
            return False
        else:
            raise error

def validate_models_access(model_ids):
    """Validates access to list of model ids using Converse API.

    Returns an empty list if all models are accessible and a list of inaccessible models otherwise.
    """
    return [model_id for model_id in model_ids if not validate_model_access(model_id)]