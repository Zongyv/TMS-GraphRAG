from nano_graphrag import GraphRAG, QueryParam
import os
from sentence_transformers import SentenceTransformer
from nano_graphrag._utils import wrap_embedding_func_with_attrs
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
import glob
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import APIError, APITimeoutError, InternalServerError, RateLimitError
import random
from typing import List, Dict, Tuple, Optional, Any, Union, Coroutine
import time
import asyncio


class TokenBucket:
    """令牌桶限流器"""

    def __init__(self, rate: float, capacity: int):
        """
        初始化令牌桶

        Args:
            rate: 令牌生成速率（每秒）
            capacity: 桶容量（最大令牌数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        尝试获取指定数量的令牌

        Args:
            tokens: 需要的令牌数量

        Returns:
            bool: 是否成功获取令牌
        """
        async with self.lock:
            now = time.time()
            # 计算从上次获取到现在新生成的令牌
            new_tokens = (now - self.last_time) * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_time = now

            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        等待直到有足够的令牌可用

        Args:
            tokens: 需要的令牌数量
        """
        while True:
            if await self.acquire(tokens):
                return
            # 计算需要等待的时间
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(max(0.01, wait_time))


# 从环境变量读取API配置
# 在运行前设置环境变量，例如：
#   export LLM_API_KEY="sk-xxx"
#   export LLM_BASE_URL="https://api.example.com/v1/"
#   export LLM_MODEL="deepseek-v3"
# 或者复制 .env.example 为 .env 文件

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def load_api_configs() -> list:
    """从环境变量加载API配置，避免硬编码密钥"""
    configs = []

    # 主要API配置（必须）
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/")
    primary_model = os.environ.get("LLM_MODEL", "deepseek-v4")
    cheap_model = os.environ.get("LLM_CHEAP_MODEL", primary_model)

    if api_key:
        configs.append({
            "name": os.environ.get("LLM_NAME", "primary"),
            "api_key": api_key,
            "base_url": base_url,
            "models": {
                "primary": primary_model,
                "cheap": cheap_model
            }
        })

    # 可选的备用API配置
    backup_key = os.environ.get("LLM_BACKUP_API_KEY", "")
    if backup_key:
        configs.append({
            "name": "backup",
            "api_key": backup_key,
            "base_url": os.environ.get("LLM_BACKUP_BASE_URL", base_url),
            "models": {
                "primary": os.environ.get("LLM_BACKUP_MODEL", primary_model),
                "cheap": os.environ.get("LLM_BACKUP_CHEAP_MODEL", cheap_model)
            }
        })

    if not configs:
        raise ValueError(
            "未找到API配置。请设置环境变量 LLM_API_KEY 和 LLM_BASE_URL，\n"
            "或在 test.py 的 load_api_configs() 函数中直接写入配置"
        )

    return configs


API_CONFIGS = load_api_configs()

# 跟踪API健康状态
api_health_status = {config["name"]: {"healthy": True, "failures": 0, "last_failure": 0} for config in API_CONFIGS}

# 为阿里云百炼模型创建专用限流器
# 根据阿里云文档调整这些参数
DASHSCOPE_QPS = 50  # 每秒查询数（QPS）限制
DASHSCOPE_TPM = 100000  # 每分钟令牌数（TPM）限制

# 创建QPS限流器（每秒请求数限制）
dashscope_qps_limiter = TokenBucket(rate=DASHSCOPE_QPS, capacity=DASHSCOPE_QPS)

# 创建TPM限流器（每分钟令牌数限制）
# 将TPM转换为每秒令牌数
dashscope_tpm_limiter = TokenBucket(rate=DASHSCOPE_TPM / 60, capacity=DASHSCOPE_TPM)

# 为不同的API端点创建限流器
api_limiters = {
    "百炼主要": dashscope_qps_limiter,
    "百炼备用1": dashscope_qps_limiter,
    "百炼备用2": dashscope_qps_limiter,
}


async def get_healthy_api_config() -> Dict:
    """获取一个健康的API配置"""
    # 重置超过5分钟未使用的API的健康状态
    current_time = time.time()
    for name, status in api_health_status.items():
        if not status["healthy"] and (current_time - status["last_failure"]) > 300:
            print(f"重置API {name}的健康状态")
            status["healthy"] = True
            status["failures"] = 0

    # 首先尝试使用主要API
    if api_health_status[API_CONFIGS[0]["name"]]["healthy"]:
        return API_CONFIGS[0]

    # 如果主要API不健康，选择一个健康的备用API
    healthy_backups = [config for config in API_CONFIGS[1:]
                       if api_health_status[config["name"]]["healthy"]]

    if healthy_backups:
        selected = random.choice(healthy_backups)
        print(f"使用备用API: {selected['name']}")
        return selected

    # 如果没有健康的备用API，使用失败次数最少的API
    least_failures = min(API_CONFIGS, key=lambda c: api_health_status[c["name"]]["failures"])
    print(f"所有API都不健康，使用失败次数最少的API: {least_failures['name']}")
    return least_failures


def mark_api_unhealthy(api_name: str):
    """标记API为不健康"""
    if api_name in api_health_status:
        api_health_status[api_name]["healthy"] = False
        api_health_status[api_name]["failures"] += 1
        api_health_status[api_name]["last_failure"] = time.time()
        print(f"API {api_name} 标记为不健康，失败次数: {api_health_status[api_name]['failures']}")


WORKING_DIR = "./workspace/rTMS"
EMBED_MODEL_DIR = "./models/embedding_model"

# 中文embedding
# EMBED_MODEL = SentenceTransformer(
#     "DMetaSoul/Dmeta-embedding-zh-small", cache_folder=EMBED_MODEL_DIR, device="cpu"
# )

# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

EMBED_MODEL = SentenceTransformer(
    "BAAI/bge-base-en-v1.5", cache_folder=EMBED_MODEL_DIR, device=device
)

# 创建一个信号量来限制并发请求
api_semaphore = asyncio.Semaphore(3)  # 最多3个并发请求


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((APIError, APITimeoutError, InternalServerError, RateLimitError)),
)
async def resilient_model_call(
        prompt, system_prompt=None, history_messages=[], return_tokens=False, **kwargs
) -> Union[tuple[Any, int], Any]:
    """带重试机制的模型调用，支持返回token消耗"""
    # 使用信号量限制并发
    async with api_semaphore:
        last_exception = None

        # 尝试所有可能的API配置
        for _ in range(1):  # 最多尝试4次不同的API
            try:
                api_config = await get_healthy_api_config()
                api_name = api_config["name"]
                print(f"使用API: {api_name}")

                # 应用API特定的限流
                if api_name in api_limiters:
                    print(f"等待 {api_name} 限流器...")
                    await api_limiters[api_name].wait_for_tokens()

                # 如果是阿里云百炼模型，还需要应用TPM限流
                if "DashScope" in api_name:
                    # 估算此次请求的令牌消耗（根据输入长度）
                    # 这里简单估算，实际应根据阿里云计费规则调整
                    input_tokens = len(prompt) // 4  # 粗略估计：4个字符约等于1个token
                    await dashscope_tpm_limiter.wait_for_tokens(input_tokens)

                openai_async_client = AsyncOpenAI(
                    api_key=api_config["api_key"],
                    base_url=api_config["base_url"],
                    timeout=60.0
                )

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
                messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                model = api_config["models"]["primary"]

                if hashing_kv is not None:
                    args_hash = compute_args_hash(model, messages)
                    if_cache_return = await hashing_kv.get_by_id(args_hash)
                    if if_cache_return is not None:
                        return if_cache_return["return"]

                response = await openai_async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )

                if hashing_kv is not None:
                    await hashing_kv.upsert(
                        {args_hash: {"return": response.choices[0].message.content, "model": model}}
                    )

                if return_tokens:
                    return response.choices[0].message.content, response.usage.total_tokens
                else:
                    return response.choices[0].message.content

            except (APITimeoutError, APIError, InternalServerError, RateLimitError) as e:
                last_exception = e
                mark_api_unhealthy(api_config["name"])
                print(f"API {api_config['name']} 调用失败: {str(e)}，尝试下一个API")
                await asyncio.sleep(2)
                continue

            except Exception as e:
                print(f"未预期的错误: {str(e)}")
                raise

        # 如果所有API都失败，抛出最后一个异常
        if last_exception:
            print("所有API尝试都失败")
            raise last_exception

        raise Exception("所有API尝试都失败，但没有捕获到具体异常")


async def resilient_cheap_model_call(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """具有弹性的廉价模型调用，带有限流"""
    # 使用信号量限制并发
    async with api_semaphore:
        last_exception = None

        for _ in range(3):
            try:
                api_config = await get_healthy_api_config()
                api_name = api_config["name"]
                print(f"使用廉价API: {api_name}")

                # 应用API特定的限流
                if api_name in api_limiters:
                    print(f"等待 {api_name} 限流器...")
                    await api_limiters[api_name].wait_for_tokens()

                # 如果是阿里云百炼模型，还需要应用TPM限流
                if "DashScope" in api_name:
                    # 估算此次请求的令牌消耗
                    input_tokens = len(prompt) // 4
                    await dashscope_tpm_limiter.wait_for_tokens(input_tokens)

                openai_async_client = AsyncOpenAI(
                    api_key=api_config["api_key"],
                    base_url=api_config["base_url"],
                    timeout=60.0
                )

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
                messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                model = api_config["models"]["cheap"]

                if hashing_kv is not None:
                    args_hash = compute_args_hash(model, messages)
                    if_cache_return = await hashing_kv.get_by_id(args_hash)
                    if if_cache_return is not None:
                        return if_cache_return["return"]

                response = await openai_async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )

                if hashing_kv is not None:
                    await hashing_kv.upsert(
                        {args_hash: {"return": response.choices[0].message.content, "model": model}}
                    )

                return response.choices[0].message.content

            except (APITimeoutError, APIError, InternalServerError, RateLimitError) as e:
                last_exception = e
                mark_api_unhealthy(api_config["name"])
                print(f"API {api_config['name']} 调用失败: {str(e)}")
                await asyncio.sleep(2)
                continue

            except Exception as e:
                print(f"未预期的错误: {str(e)}")
                raise

        if last_exception:
            raise last_exception

        raise Exception("所有API尝试都失败")


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

if __name__ == "__main__":
    graph_func = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=local_embedding,
        best_model_func=resilient_model_call,
        cheap_model_func=resilient_cheap_model_call,
        embedding_batch_num=64,
        enable_llm_cache=False
    )

    md_dir = "D:/YJS/TMSrag/rTMS-rag/process/dataset/rTMS/markdown/*"
    file_path_list = glob.glob(md_dir + "/*-with-image-refs.md")
    print(f"找到 {len(file_path_list)} 个MD文件")
    for file_path in file_path_list:
        print(f"处理文件: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            graph_func.insert(f.read())

    # # Perform global graphrag search
    # print(graph_func.query("What are the top themes in this story?"))

    # # Perform local graphrag search (I think is better and more scalable one)
    # print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))

    # print(
    #     graph_func.query(
    #         "What are the TMS stimulation sites for depression, Alzheimer's disease, Parkinson's disease and multiple sclerosis respectively? What are the connections between the stimulus sites?，请结合知识图谱，用简洁的语言回答",
    #         param=QueryParam(mode="local")
    #     )
    # )