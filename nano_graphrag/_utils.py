import asyncio
import html
import json
import logging
import os
import re
import numbers
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union

import numpy as np
import tiktoken

# 日志记录器
logger = logging.getLogger("nano-graphrag")
logging.getLogger("neo4j").setLevel(logging.ERROR)
ENCODER = None

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
        确保在任何情况下都能获取到一个事件循环。
        
        这个函数会尝试获取当前线程中的事件循环，如果当前线程中没有事件循环，则会创建一个新的。
    """
    try:
        # 如果当前线程中已经存在事件循环，则使用它。
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果当前线程中没有事件循环，则创建一个新的。
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_first_complete_json(s: str):
    """
        从字符串中提取第一个完整的JSON对象，使用栈来跟踪花括号。
        
        这个函数会遍历输入字符串，使用一个栈来跟踪花括号的位置。
        当遇到一个左花括号时，将其索引压入栈中。
        当遇到一个右花括号时，从栈中弹出一个索引，并检查栈是否为空。
        如果栈为空，则表示当前右花括号是一个完整的JSON对象的结束标记。
        
        如果栈不为空，则继续遍历字符串，直到找到一个完整的JSON对象。
        
        参数:
            s: 输入的字符串。
        返回:
            json.loads(first_json_str.replace("\n", "")): 提取的第一个完整的JSON对象。
            None: 如果未找到完整的JSON对象。
    """ 
    stack = []
    first_json_start = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start:i+1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None

def parse_value(value: str):
    """
        将字符串值转换为适当的类型（int, float, bool, None, 或保持为字符串）。
        
        这个函数会遍历输入字符串，并根据其内容尝试将其转换为适当的类型。

        因为LLM的输出不会是相应的类型，只能是字符串，所以需要这个函数来将字符串转换为适当的类型。
        
        参数:
            value: 输入的字符串。
        返回:
            value.strip('"'): 转换后的值。
    """
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # 如果字符串中包含点，则可能是一个浮点数
                return float(value)
            else:
                return int(value)
        except ValueError:
            # 如果转换失败，则返回字符串（可能是一个字符串）
            return value.strip('"')  # 移除周围的引号（如果存在）

def extract_values_from_json(json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False):
    """
        从非标准或格式不正确的JSON字符串中提取键值，处理嵌套对象。
    """
    extracted_values = {}
    
    # 增强的正则表达式模式，匹配带引号和不带引号的值，以及嵌套的对象
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'
    
    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # 移除键周围的引号
        value = match.group('value').strip()

        # 如果值是另一个嵌套的JSON（以'{'开头并以'}'结尾），则递归解析它
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # 将值转换为适当的类型（int, float, bool, etc.）
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    
    return extracted_values


def convert_response_to_json(response: str) -> dict:
    """
        将响应字符串转换为JSON，具有错误处理和回退到非标准JSON提取。
    """
    prediction_json = extract_first_complete_json(response)
    
    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)
    
    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")
    
    return prediction_json




def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """
        使用tiktoken对字符串进行编码并计算token数量。 
        tokenlize by openai's tiktoken
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    """
        根据token大小截断列表数据。
        该函数的目的是确保列表中数据的总token数不超过指定的最大token大小。
        当数据的总token数超过最大允许大小时，函数将返回截断后的列表。
        参数:
            - list_data: list, 需要截断的列表，其中每个元素为一个数据项。
            - key: callable, 用于从列表数据项中提取用于计算token大小的字符串的函数。
            - max_token_size: int, 允许的最大token大小，用于决定列表数据的截断点。
        返回:
            - 截断后的列表。如果max_token_size小于等于0，返回空列表。
        注意:
            - 该函数使用tiktoken对字符串进行编码并计算token数量，请确保在使用前已安装tiktoken库。
            - 截断操作基于累计token数量首次超过max_token_size发生的索引位置。
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    """
        计算内容的MD5哈希值。
        
        该函数使用MD5哈希算法对输入内容进行哈希计算，并返回哈希值。
        
        参数:
            content: 需要计算哈希值的内容。
            prefix: 可选的前缀字符串，用于在哈希值前面添加。
        返回:
            prefix + md5(content.encode()).hexdigest(): 计算得到的MD5哈希值。
    """
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(prompt: str, generated_content: str, using_amazon_bedrock: bool):
    """
        将用户和助手的对话打包为OpenAI消息格式。
        该函数接受一系列字符串参数，成对地将它们包装成交替的用户和助手角色的消息。
        这对于将对话历史记录转换为可供OpenAI的API处理的格式特别有用。在_op.py里就是调用给history的。
        参数:
            *args (str): 一个或多个字符串参数，表示用户和助手之间的对话交替发言。
        返回:
            list: 一个字典列表，每个字典包含两个键值对:
                - 'role': 表示消息发送者的角色，根据参数序列中的位置交替为'user'或'assistant'。
                - 'content': 发送者发送的消息内容，来自输入参数序列中的对应位置。
    """
    if using_amazon_bedrock:
        return [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": generated_content}]},
        ]
    else:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": generated_content},
        ]


def is_float_regex(value):
    """
        检查字符串是否表示一个浮点数。
        
        该函数使用正则表达式来验证输入字符串是否符合浮点数的格式。
        
        参数:
            value: 需要检查的字符串。
        返回:
            bool: 如果字符串表示一个浮点数，则返回True，否则返回False。
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    """
        计算参数的哈希值。
        
        该函数将输入参数转换为字符串并计算其MD5哈希值。
        
        参数:
            *args: 一个或多个参数，可以是任何类型。
        返回:
            md5(str(args).encode()).hexdigest(): 计算得到的MD5哈希值。
    """
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """
        根据多个标记分割字符串。
        
        该函数接受一个字符串和一组标记，将字符串按这些标记分割成多个子字符串，并返回一个包含这些子字符串的列表。
        
        参数:
            content: 需要分割的字符串。
            markers: 一组标记，用于分割字符串。
        返回:
            list: 一个包含分割后子字符串的列表。
    """
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    """将多维列表转换为CSV格式，用在社区结构部分"""
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""
    """
        限制异步函数调用的最大次数。
        
        该装饰器接受一个最大调用次数和一个等待时间，并返回一个装饰器函数。
        
        参数:
            max_size: 最大调用次数。
            waitting_time: 等待时间。
        返回:
            final_decro: 装饰器函数。
    """
    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""
    """
        包装一个函数，并添加属性。
        
        该装饰器接受一个或多个关键字参数，并返回一个装饰器函数。
        
        参数:
            **kwargs: 一个或多个关键字参数，用于包装函数。
        返回:
            final_decro: 装饰器函数。
    """
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
