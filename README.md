# Instruction Extraction

## Environment Preparation

```bash
conda install python-dotenv
```

## Llama 3.1

The key "meta-llama/Meta-Llama-3.1-8B" for api key.

## Llama 3

The key "meta-llama/Meta-Llama-3-70B-Instruct" for api key.

## Image serving

```bash
openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes
python https_server.py
```

## Call Open Llama 3.2-70B

1. Set up for "Github key"
refer to GITHUB_TOKEN in .env local file

2. Install dependency

```bash
pip install azure-ai-inference
```

About the public API, refer to

1. github api - <https://github.com/marketplace/models/catalog>
2. novita.ai api - <https://novita.ai/model-api/product/llm-api/playground>