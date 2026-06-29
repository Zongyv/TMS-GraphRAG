import re
import json
import asyncio
import tiktoken
from typing import Union, List, Any
from collections import Counter, defaultdict
from nano_graphrag._splitter import SeparatorSplitter
from nano_graphrag._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from nano_graphrag.prompt import GRAPH_FIELD_SEP, PROMPTS
import requests
import os

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results



def search_crossref_by_title(title, max_results=5):
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": title,
        "rows": max_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        items = response.json().get("message", {}).get("items", [])
        results = []
        for item in items:
            results.append({
                "title": item.get("title", [""])[0],
                "DOI": item.get("DOI", ""),
                "URL": item.get("URL", ""),
                "author": ", ".join([f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])]),
                "published": item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            })
        return results
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return []


def remove_special_characters(text: str) -> str:
    # 保留中文、英文、数字，空格，其余全部移除
    return re.sub(r"[^\u4e00-\u9fff\sA-Za-z0-9\u3000]", "", text)

def remove_markdown_images(text):
    # 匹配 ![任何文字](任何链接)
    return re.sub(r'!\[.*?\]\(.*?\)', '', text)

def remove_all_whitespace(text: str) -> str:
    # 移除所有空白字符
    return re.sub(r"\s+|\u3000", "", text)

def chunking_by_markdown_with_filter(
    tokens_list: list[list[int]],
    doc_keys: list[Any],
    tiktoken_model,
    del_headers: list[str] = [
        "References", 
        "Acknowledgments", 
        "Appendix",
        "AUTHOR CONTRIBUTIONS",
        "FUNDING",
        "ARTICLE INFORMATION",
        "Supplementary Material",
        "Supplementary Tables",
        "Supplementary Figures",
        "Supplementary Data",
        "Supplementary Information",
        "Supplementary Material",
        "Supplementary Tables",
        "Supplementary Figures",
        "Potential Conflicts of Interest"
        ],  # 移除的标题列表
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    absolute_max_token_size: int = 1200,  # 添加绝对最大token限制
):
    """
    按 Markdown 标题结构分块，并只保留以特定标题开头的块。
    确保每个chunk不超过absolute_max_token_size。
    """
    results = []

    header_pattern = re.compile(r'(#{1,6}\s+[^\n]+)')  # 匹配 Markdown 标题

    for idx, tokens in enumerate(tokens_list):
        full_text = tiktoken_model.decode(tokens)
        full_text = remove_markdown_images(full_text)
        parts = header_pattern.split(full_text)

        segments = []
        headers = []
        for i in range(1, len(parts), 2):
            header = parts[i]
            header_without_special_characters = remove_special_characters(header).strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            segment_text = header + body
            if not any(remove_all_whitespace(header_without_special_characters).upper().startswith(remove_all_whitespace(h).upper()) for h in del_headers):
                segments.append(segment_text)
                headers.append(header_without_special_characters)
            if remove_all_whitespace(header_without_special_characters).upper().startswith("REFERENCES".upper()):
                break
        if not segments:
            continue  # 没有符合的段落，跳过此文档

        # 寻找文章标题
        for header in headers[:3]:
            if len(header) > 20:
                title = search_crossref_by_title(header)
                for t in title:
                    if t["title"].startswith(header):
                        segments[0] = "[[article]]" + t["title"] + "[[/article]]\n" + segments[0]
                        break

        # 将每个段落编码为token
        seg_tokens = [tiktoken_model.encode(seg) for seg in segments]
        
        # 处理超长段落，将其分割成更小的部分
        processed_seg_tokens = []
        for seg in seg_tokens:
            if len(seg) > absolute_max_token_size:
                # 如果单个段落超过最大限制，将其分割成多个小段落
                for start in range(0, len(seg), absolute_max_token_size - overlap_token_size):
                    end = min(start + absolute_max_token_size, len(seg))
                    processed_seg_tokens.append(seg[start:end])
            else:
                processed_seg_tokens.append(seg)
        
        # 按 token 数组进行重组并考虑 overlap
        chunks: List[List[int]] = []
        current_chunk: List[int] = []
        for seg in processed_seg_tokens:
            # 检查添加当前段落是否会超过最大限制
            if len(current_chunk) + len(seg) <= max_token_size:
                current_chunk.extend(seg)
            else:
                # 如果当前chunk已经有内容，保存它
                if current_chunk:
                    chunks.append(current_chunk)
                    # 保留overlap部分
                    overlap_tokens = current_chunk[-overlap_token_size:] if len(current_chunk) > overlap_token_size else current_chunk
                else:
                    overlap_tokens = []
                
                # 检查段落本身是否超过最大限制
                if len(seg) > absolute_max_token_size - len(overlap_tokens):
                    # 如果段落太长，只取能放入的部分
                    current_chunk = overlap_tokens + seg[:absolute_max_token_size - len(overlap_tokens)]
                    chunks.append(current_chunk)
                    
                    # 处理剩余部分
                    remaining = seg[absolute_max_token_size - len(overlap_tokens):]
                    while remaining:
                        current_chunk = remaining[:absolute_max_token_size]
                        if len(current_chunk) == absolute_max_token_size:
                            chunks.append(current_chunk)
                            remaining = remaining[absolute_max_token_size - overlap_token_size:]
                        else:
                            current_chunk = remaining
                            remaining = []
                else:
                    current_chunk = overlap_tokens + seg
        
        # 添加最后一个chunk
        if current_chunk and len(current_chunk) <= absolute_max_token_size:
            chunks.append(current_chunk)
        elif current_chunk:
            # 如果最后一个chunk太长，分割它
            for start in range(0, len(current_chunk), absolute_max_token_size - overlap_token_size):
                end = min(start + absolute_max_token_size, len(current_chunk))
                chunks.append(current_chunk[start:end])

        # 输出 chunk 结果
        lengths = [len(chunk) for chunk in chunks]
        texts = tiktoken_model.decode_batch(chunks)
        for i, text in enumerate(texts):
            results.append({
                "tokens": lengths[i],
                "content": text.strip(),
                "chunk_order_index": i,
                "full_doc_id": doc_keys[idx],
            })

    return results


def chunking_by_markdown_with_table_preservation(
    tokens_list: list[list[int]],
    doc_keys: list[Any],
    tiktoken_model,
    del_headers: list[str] = [
        "References",
        "Appendix",
        "AUTHOR CONTRIBUTIONS",
        "ARTICLE INFORMATION",
        "Supplementary Material",
        "Supplementary Tables",
        "Supplementary Figures",
        "Supplementary Data",
        "Supplementary Information",
        "Potential Conflicts of Interest"
    ],
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    absolute_max_token_size: int = 1200,
):
    """
    结合表格保护和Markdown过滤的分块方法。
    表格保持完整，非表格部分按Markdown标题结构分块并过滤不需要的章节。
    """
    results = []

    for idx, tokens in enumerate(tokens_list):
        full_text = tiktoken_model.decode(tokens)
        full_text = remove_markdown_images(full_text)
        
        # 识别表格边界
        table_boundaries = _find_table_boundaries(full_text)
        
        # 根据表格边界分割文本
        text_segments = _split_text_by_tables(full_text, table_boundaries)
        
        chunk_order_index = 0
        
        for segment in text_segments:
            if segment['type'] == 'table':
                # 表格直接作为一个chunk
                table_tokens = tiktoken_model.encode(segment['content'])
                if len(table_tokens) > max_token_size:
                    print(f"警告: 表格超过token限制 ({len(table_tokens)} > {max_token_size})，但保持完整")
                
                results.append({
                    "tokens": len(table_tokens),
                    "content": segment['content'].strip(),
                    "chunk_order_index": chunk_order_index,
                    "full_doc_id": doc_keys[idx],
                    "contains_table": True
                })
                chunk_order_index += 1
                
            else:  # 非表格文本
                # 使用chunking_by_markdown_with_filter的逻辑
                filtered_chunks = _process_non_table_text_with_filter(
                    segment['content'],
                    del_headers,
                    tiktoken_model,
                    max_token_size,
                    overlap_token_size,
                    absolute_max_token_size
                )
                
                for chunk_content in filtered_chunks:
                    chunk_tokens = tiktoken_model.encode(chunk_content)
                    results.append({
                        "tokens": len(chunk_tokens),
                        "content": chunk_content.strip(),
                        "chunk_order_index": chunk_order_index,
                        "full_doc_id": doc_keys[idx],
                        "contains_table": False
                    })
                    chunk_order_index += 1

    return results


def _split_text_by_tables(text: str, table_boundaries: list[tuple[int, int]]) -> list[dict]:
    """
    根据表格边界分割文本，返回文本段和表格段
    """
    segments = []
    current_pos = 0
    
    for table_start, table_end in table_boundaries:
        # 添加表格前的文本段
        if current_pos < table_start:
            text_content = text[current_pos:table_start].strip()
            if text_content:
                segments.append({
                    'type': 'text',
                    'content': text_content
                })
        
        # 添加表格段
        table_content = text[table_start:table_end].strip()
        if table_content:
            segments.append({
                'type': 'table',
                'content': table_content
            })
        
        current_pos = table_end
    
    # 添加最后一个表格后的文本
    if current_pos < len(text):
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            segments.append({
                'type': 'text',
                'content': remaining_text
            })
    
    return segments

def _process_non_table_text_with_filter(
    text: str,
    del_headers: list[str],
    tiktoken_model,
    max_token_size: int,
    overlap_token_size: int,
    absolute_max_token_size: int
) -> list[str]:
    """
    对非表格文本使用chunking_by_markdown_with_filter的逻辑
    """
    header_pattern = re.compile(r'(#{1,6}\s+[^\n]+)')
    parts = header_pattern.split(text)
    
    segments = []
    headers = []
    
    # 处理第一个部分（可能没有标题的文本）
    if len(parts) > 0 and parts[0].strip():
        segments.append(parts[0].strip())
        headers.append("")  # 无标题
    
    # 处理有标题的部分
    for i in range(1, len(parts), 2):
        header = parts[i]
        header_without_special_characters = remove_special_characters(header).strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        segment_text = header + body
        
        # 过滤不需要的章节
        if not any(remove_all_whitespace(header_without_special_characters).upper().startswith(remove_all_whitespace(h).upper()) for h in del_headers):
            segments.append(segment_text)
            headers.append(header_without_special_characters)
        
        # 遇到References就停止
        if remove_all_whitespace(header_without_special_characters).upper().startswith("REFERENCES".upper()):
            break
    
    if not segments:
        return []
    
    # 寻找文章标题并添加到第一个segment（只对有标题的segment）
    for i, header in enumerate(headers[:3]):
        if header and len(header) > 20:
            title = search_crossref_by_title(header)
            for t in title:
                if t["title"].startswith(header):
                    segments[i] = "[[article]]" + t["title"] + "[[/article]]\n" + segments[i]
                    break
            break
    
    # 将segments编码为tokens并进行分块
    seg_tokens = [tiktoken_model.encode(seg) for seg in segments]
    
    # 处理超长段落
    processed_seg_tokens = []
    for seg in seg_tokens:
        if len(seg) > absolute_max_token_size:
            for start in range(0, len(seg), absolute_max_token_size - overlap_token_size):
                end = min(start + absolute_max_token_size, len(seg))
                processed_seg_tokens.append(seg[start:end])
        else:
            processed_seg_tokens.append(seg)
    
    # 重组chunks
    chunks: List[List[int]] = []
    current_chunk: List[int] = []
    
    for seg in processed_seg_tokens:
        if len(current_chunk) + len(seg) <= max_token_size:
            current_chunk.extend(seg)
        else:
            if current_chunk:
                chunks.append(current_chunk)
                overlap_tokens = current_chunk[-overlap_token_size:] if len(current_chunk) > overlap_token_size else current_chunk
            else:
                overlap_tokens = []
            
            if len(seg) > absolute_max_token_size - len(overlap_tokens):
                current_chunk = overlap_tokens + seg[:absolute_max_token_size - len(overlap_tokens)]
                chunks.append(current_chunk)
                
                remaining = seg[absolute_max_token_size - len(overlap_tokens):]
                while remaining:
                    current_chunk = remaining[:absolute_max_token_size]
                    if len(current_chunk) == absolute_max_token_size:
                        chunks.append(current_chunk)
                        remaining = remaining[absolute_max_token_size - overlap_token_size:]
                    else:
                        current_chunk = remaining
                        remaining = []
            else:
                current_chunk = overlap_tokens + seg
    
    # 添加最后一个chunk
    if current_chunk and len(current_chunk) <= absolute_max_token_size:
        chunks.append(current_chunk)
    elif current_chunk:
        for start in range(0, len(current_chunk), absolute_max_token_size - overlap_token_size):
            end = min(start + absolute_max_token_size, len(current_chunk))
            chunks.append(current_chunk[start:end])
    
    # 解码为文本
    return tiktoken_model.decode_batch(chunks)


def _find_table_boundaries(text: str) -> list[tuple[int, int]]:
    """
    找到文本中所有表格的边界位置
    返回: [(start_pos, end_pos), ...]
    """
    import re

    table_boundaries = []

    # 匹配Markdown表格模式
    # 表格通常以 |---|---| 这样的分隔行标识
    table_pattern = re.compile(
        r'(\|[^\n]*\|[\s]*\n\|[\s]*[-:]+[\s]*\|[^\n]*\n(?:\|[^\n]*\|[\s]*\n)*)',
        re.MULTILINE
    )

    for match in table_pattern.finditer(text):
        start_pos = match.start()
        end_pos = match.end()

        # 扩展到完整的表格（包括表格前后的标题和说明）
        extended_start, extended_end = _extend_table_boundaries(text, start_pos, end_pos)
        table_boundaries.append((extended_start, extended_end))

    # 合并重叠的表格边界
    return _merge_overlapping_boundaries(table_boundaries)


def _extend_table_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """
    扩展表格边界，包含表格标题和说明
    """
    lines = text.split('\n')

    # 找到表格开始和结束的行号
    start_line = text[:start].count('\n')
    end_line = text[:end].count('\n')

    # 向前查找表格标题（通常在表格前1-2行）
    extended_start_line = start_line
    for i in range(max(0, start_line - 3), start_line):
        line = lines[i].strip()
        if line and (line.startswith('#') or 'table' in line.lower() or '表' in line):
            extended_start_line = i
            break

    # 向后查找表格说明（通常在表格后1-2行）
    extended_end_line = end_line
    for i in range(end_line, min(len(lines), end_line + 3)):
        line = lines[i].strip()
        if line and (line.startswith('Note:') or line.startswith('注:') or 'caption' in line.lower()):
            extended_end_line = i + 1
        elif not line:  # 空行表示表格结束
            break

    # 转换回字符位置
    extended_start = sum(len(lines[i]) + 1 for i in range(extended_start_line))
    extended_end = sum(len(lines[i]) + 1 for i in range(extended_end_line))

    return extended_start, min(extended_end, len(text))


def _merge_overlapping_boundaries(boundaries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    合并重叠的边界
    """
    if not boundaries:
        return []

    sorted_boundaries = sorted(boundaries)
    merged = [sorted_boundaries[0]]

    for start, end in sorted_boundaries[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # 重叠
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _split_text_preserving_tables(
        text: str,
        table_boundaries: list[tuple[int, int]],
        tiktoken_model,
        max_token_size: int,
        overlap_token_size: int
) -> list[str]:
    """
    分割文本，保持表格完整
    """
    chunks = []
    current_pos = 0

    for table_start, table_end in table_boundaries:
        # 处理表格前的文本
        if current_pos < table_start:
            pre_table_text = text[current_pos:table_start].strip()
            if pre_table_text:
                # 对非表格文本进行常规分块
                pre_chunks = _chunk_regular_text(
                    pre_table_text,
                    tiktoken_model,
                    max_token_size,
                    overlap_token_size
                )
                chunks.extend(pre_chunks)

        # 处理表格（作为单独的chunk，无视token限制）
        table_text = text[table_start:table_end].strip()
        if table_text:
            table_tokens = len(tiktoken_model.encode(table_text))
            if table_tokens > max_token_size:
                print(f"警告: 表格超过token限制 ({table_tokens} > {max_token_size})，但保持完整")
            chunks.append(table_text)

        current_pos = table_end

    # 处理最后一个表格后的文本
    if current_pos < len(text):
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            remaining_chunks = _chunk_regular_text(
                remaining_text,
                tiktoken_model,
                max_token_size,
                overlap_token_size
            )
            chunks.extend(remaining_chunks)

    return chunks


def _chunk_regular_text(
        text: str,
        tiktoken_model,
        max_token_size: int,
        overlap_token_size: int
) -> list[str]:
    """
    对常规文本进行分块
    """
    tokens = tiktoken_model.encode(text)
    chunks = []

    for start in range(0, len(tokens), max_token_size - overlap_token_size):
        chunk_tokens = tokens[start:start + max_token_size]
        chunk_text = tiktoken_model.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def _contains_table(text: str) -> bool:
    """
    检查文本是否包含表格
    """
    import re
    table_pattern = re.compile(r'\|[^\n]*\|[\s]*\n\|[\s]*[-:]+[\s]*\|')
    return bool(table_pattern.search(text))



def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


def get_meta_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]  # 这里就是DOI

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        # 计算chunk在原文中的字符位置
        original_content = new_docs[chunk["full_doc_id"]]["content"]
        chunk_content = chunk["content"]

        # 查找chunk在原文中的起始位置
        start_position = original_content.find(chunk_content.strip())
        if start_position == -1:
            # 如果找不到精确匹配，尝试查找部分匹配
            chunk_words = chunk_content.strip().split()[:10]  # 取前10个词
            search_text = " ".join(chunk_words)
            start_position = original_content.find(search_text)

        end_position = start_position + len(chunk_content) if start_position != -1 else -1

        # 添加DOI和位置信息到chunk
        enhanced_chunk = {
            **chunk,
            "paper_doi": chunk["full_doc_id"],  # DOI信息
            "start_position": start_position,  # 在原文中的起始字符位置
            "end_position": end_position,  # 在原文中的结束字符位置
            "position_info": {
                "char_start": start_position,
                "char_end": end_position,
                "chunk_length": len(chunk_content)
            }
        }

        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): enhanced_chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    using_amazon_bedrock: bool=False,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]

        chunk_record = {
            "chunk_key": chunk_key,
            "prompt": content,
            "response": final_result
        }
        record_dir = global_config["working_dir"]
        record_file = os.path.join(record_dir, f"chunk_record.json")
        with open(record_file, "a", encoding="utf-8") as f:
            json.dump(chunk_record, f, ensure_ascii=False, indent=2)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result, using_amazon_bedrock)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    node_degrees = await knwoledge_graph_inst.node_degrees_batch(nodes_in_order)
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            node_degrees[i],
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edge_degrees = await knwoledge_graph_inst.edge_degrees_batch(edges_in_order)
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            edge_degrees[i]
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)

        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await knowledge_graph_inst.get_nodes_batch(all_one_hop_nodes)
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    
    all_edges = []
    seen = set()
    
    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge) 
                
    all_edges_pack = await knowledge_graph_inst.get_edges_batch(all_edges)
    all_edges_degree = await knowledge_graph_inst.edge_degrees_batch(all_edges)
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await knowledge_graph_inst.get_nodes_batch([r["entity_name"] for r in results])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await knowledge_graph_inst.node_degrees_batch([r["entity_name"] for r in results])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
    )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response
