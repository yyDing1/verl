# internlm/internlm2-1_8b-reward
# vllm serve internlm/internlm2-1_8b-reward

# payload = {
#     "input": "Hello, my name is"
# }
# import requests
# response = requests.post("http://localhost:8000/classify", json=payload)
# print(response.json())
# 


# CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path Skywork/Skywork-Reward-V2-Llama-3.2-1B --port 11111
# CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path Skywork/Skywork-Reward-V2-Llama-3.2-1B --port 22222
# python -m sglang_router.launch_router --worker-urls http://localhost:11111 http://localhost:22222
payload = {
    "input": "Hello, my name is",
}
import requests
response = requests.post("http://127.0.0.1:30000/v1/embeddings", json=payload)
print(response.json())
