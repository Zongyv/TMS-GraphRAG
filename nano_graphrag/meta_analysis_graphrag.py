import asyncio
import glob
import os
from typing import List, Dict, Optional, Tuple, Type, cast, Callable, Union
from dataclasses import asdict, dataclass, field
import tiktoken
import re
from nano_graphrag.prompt import PROMPTS
import time

from nano_graphrag._storage import (
    JsonKVStorage,
    NetworkXStorage,
    NanoVectorDBStorage
)
from nano_graphrag._utils import (
    logger,
    compute_mdhash_id,
    always_get_an_event_loop,
    convert_response_to_json
)
from nano_graphrag.base import (
    BaseKVStorage,
    BaseGraphStorage,
    BaseVectorStorage,
    StorageNameSpace
)
from nano_graphrag._op import (
    get_meta_chunks,
    chunking_by_markdown_with_table_preservation
)
from test import (
    local_embedding,
    resilient_model_call,
    resilient_cheap_model_call
)


@dataclass
class MetaAnalysisGraphRAG:
    """专门用于Meta分析的GraphRAG系统"""

    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_markdown_with_table_preservation
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    working_dir: str = "./meta_analysis_cache"

    # Meta分析专用配置
    quality_threshold: float = 6.0  # 论文质量阈值
    bias_risk_threshold: str = "moderate"  # 偏倚风险阈值
    min_sample_size: int = 20  # 最小样本量

    # 存储组件类型（复用GraphRAG的存储结构）
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage

    # 嵌入和模型函数
    embedding_func: callable = field(default_factory=lambda: local_embedding)
    best_model_func: callable = resilient_model_call
    cheap_model_func: callable = resilient_cheap_model_call

    # 存储组件实例（在__post_init__中初始化）
    raw_papers_storage: Optional[BaseKVStorage] = field(default=None, init=False)
    evaluated_papers_storage: Optional[BaseKVStorage] = field(default=None, init=False)
    meta_knowledge_graph: Optional[BaseGraphStorage] = field(default=None, init=False)
    evidence_vector_db: Optional[BaseVectorStorage] = field(default=None, init=False)

    # Meta分析专用函数
    paper_evaluator_func: Optional[callable] = None
    evidence_synthesizer_func: Optional[callable] = None
    incremental_assessor_func: Optional[callable] = None

    # 统计信息（在__post_init__中初始化）
    _stats: Optional[Dict] = field(default=None, init=False)

    def __post_init__(self):
        """初始化Meta分析专用存储和组件"""

        # 验证必要的函数是否提供
        if self.embedding_func is None:
            raise ValueError("embedding_func 是必需的")
        if self.best_model_func is None:
            raise ValueError("best_model_func 是必需的")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # 初始化存储组件
        global_config = asdict(self)

        # 原始论文存储
        self.raw_papers_storage = self.key_string_value_json_storage_cls(
            namespace="raw_papers",
            global_config=global_config
        )

        self.raw_papers_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        # 评估后的高质量论文存储
        self.evaluated_papers_storage = self.key_string_value_json_storage_cls(
            namespace="evaluated_papers",
            global_config=global_config
        )

        # 初始化统计信息
        self._stats = {
            "total_papers_processed": 0,
            "total_time_seconds": 0.0,
            "total_tokens_used": 0,
            "paper_times": [],  # 每篇论文的处理时间
            "paper_tokens": [],  # 每篇论文的token消耗
            "processed_paper_ids": set(),  # 新增：已处理的论文ID集合
            "start_time": None,
            "end_time": None
        }
        
        # 从历史文件加载统计数据
        self._load_statistics_from_file()

    def insert_papers(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_papers(string_or_strings))

    async def ainsert_papers(self, paper_content: str) -> None:
        """插入单篇论文并进行预处理（类似GraphRAG的insert方法）"""

        paper_start_time = time.time()
        paper_tokens = 0
        paper_id = None
        paper_title = None

        logger.info("开始处理论文...")

        try:
            inserting_chunks = await self._process_paper_chunks(paper_content)

            # 如果返回空字典，说明论文已处理过，直接返回
            if not inserting_chunks:
                logger.info("论文已处理，跳过")
                return

            # 阶段2: 基于文本块提取论文元数据
            try:
                # 获取论文ID
                first_chunk = next(iter(inserting_chunks.values()))
                paper_id = first_chunk.get("full_doc_id")

                metadata_tokens = await self._extract_paper_metadata_from_chunks(inserting_chunks)
                paper_tokens += metadata_tokens

                # 尝试从evaluated_papers_storage获取标题
                if paper_id:
                    paper_data = await self.evaluated_papers_storage.get_by_id(paper_id)
                    if paper_data:
                        paper_title = paper_data.get("title", "Unknown")

            except Exception as e:
                logger.error(f"论文元数据提取失败，清理已保存的分块数据: {str(e)}")
                await self._cleanup_failed_paper(inserting_chunks)
                raise

            logger.info("论文预处理完成")

            # 记录统计信息
            paper_end_time = time.time()
            paper_time = paper_end_time - paper_start_time

            self._stats["total_papers_processed"] += 1
            self._stats["total_time_seconds"] += paper_time
            self._stats["total_tokens_used"] += paper_tokens
            self._stats["paper_times"].append(paper_time)
            self._stats["paper_tokens"].append(paper_tokens)
            if paper_id:
                self._stats["processed_paper_ids"].add(paper_id)

            # 准备论文信息
            paper_info = {
                "paper_id": paper_id or "Unknown",
                "title": paper_title or "Unknown",
                "processing_time": paper_time,
                "tokens_used": paper_tokens
            }

            # 写入统计文件
            self._write_paper_statistics(paper_info)

            logger.info(f"本篇论文处理时间: {paper_time:.2f}秒, Token消耗: {paper_tokens}")

        except Exception as e:
            logger.error(f"论文处理失败: {str(e)}")
            raise

    async def _process_paper_chunks(self, paper_content: str) -> dict:
        """处理单篇论文分块"""
        await self._insert_start()
        try:
            logger.info("开始论文分块处理...")

            if isinstance(paper_content, str):
                paper_content = [paper_content]

            # 首先提取DOI作为论文标识符
            paper_dois = {}
            for content in paper_content:
                doi = await self._extract_doi_from_content(content)
                if doi:
                    paper_dois[doi] = {"content": content.strip()}
                else:
                    # 如果没有DOI，使用hash作为备用标识符
                    fallback_id = compute_mdhash_id(content.strip(), prefix="paper-")
                    paper_dois[fallback_id] = {"content": content.strip()}

            # 检查是否已在历史统计中处理过
            for paper_id in paper_dois.keys():
                if paper_id in self._stats["processed_paper_ids"]:
                    logger.info(f"论文 {paper_id} 已在历史记录中，跳过处理")
                    return {}

            _add_paper_keys = await self.raw_papers_storage.filter_keys(list(paper_dois.keys()))
            new_paper = {k: v for k, v in paper_dois.items() if k in _add_paper_keys}

            inserting_chunks = get_meta_chunks(
                new_docs=new_paper,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )
            _add_chunk_keys = await self.raw_papers_chunks.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return {}
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.raw_papers_storage.upsert(new_paper)
            await self.raw_papers_chunks.upsert(inserting_chunks)

            await self._insert_done()

            return inserting_chunks

        except Exception as e:
            logger.error(f"论文分块失败: {str(e)}")
            raise


    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.raw_papers_storage,
            self.raw_papers_chunks,
            self.evaluated_papers_storage
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


    async def _extract_paper_metadata_from_chunks(self, chunks: dict) -> int:
        """基于文本块提取论文的meta分析元数据，返回token消耗量"""
        logger.info("开始从文本块提取论文元数据...")
        await self._insert_start()
        
        total_tokens = 0
        
        if not chunks:
            logger.warning("没有chunks可供提取元数据")
            return 0
        
        try:
            # 获取DOI（所有chunks的DOI都相同）
            first_chunk = next(iter(chunks.values()))
            doi = first_chunk.get("full_doc_id")
            
            logger.info(f"提取论文元数据: {doi}")
            
            # 将chunks转换为列表便于处理
            chunk_list = list(chunks.values())
            
            # 按chunk_order_index排序，确保顺序正确
            chunk_list.sort(key=lambda x: x.get("chunk_order_index", 0))
            
            # 提取不同类型的chunks
            abstract_chunks = chunk_list[:5]  # 传入前5个chunks作为abstract_chunks
            methods_chunks = self._find_chunks_by_type(chunk_list, "methods")
            results_chunks = self._find_chunks_by_type(chunk_list, "results")
            table_chunks = self._find_table_chunks(chunk_list)

            study_design, tokens = await self._identify_study_design(abstract_chunks, methods_chunks)
            total_tokens = self._safe_add_tokens(total_tokens, tokens)
            logger.info(f"识别到研究设计: {study_design.get('study_design_type')}, tokens: {tokens}")

            # 从文本中提取结局指标定义
            outcome_definitions, tokens = await self._extract_outcome_definitions(methods_chunks)
            total_tokens = self._safe_add_tokens(total_tokens, tokens)

            # 使用前几个chunks提取基础元数据（包含标题、作者、摘要）
            basic_metadata, tokens = await self._extract_basic_metadata_from_chunks(
                abstract_chunks, methods_chunks
            )
            total_tokens = self._safe_add_tokens(total_tokens, tokens)
            
            # 根据结局指标定义和研究类型从表格中提取数值数据
            design_type = study_design.get("study_design_type") if study_design else "two-arm rct"

            numerical_data, tokens = await self._extract_numerical_data_from_tables(
                table_chunks, outcome_definitions, chunk_list, study_design
            )
            total_tokens = self._safe_add_tokens(total_tokens, tokens)

            if design_type == "multi-arm rct":
                numerical_data = self._impute_missing_sd_for_multi_arm(numerical_data)
            elif design_type == "double-blind sham-controlled crossover":
                numerical_data = self._impute_missing_sd_for_crossover(numerical_data)
            else:
                # 补充缺失的标准差
                if numerical_data.get("primary_outcomes"):
                    for i, outcome in enumerate(numerical_data["primary_outcomes"]):
                        if outcome:
                            numerical_data["primary_outcomes"][i] = self._impute_missing_sd(outcome)

                if numerical_data.get("secondary_outcomes"):
                    for i, outcome in enumerate(numerical_data["secondary_outcomes"]):
                        if outcome:
                            numerical_data["secondary_outcomes"][i] = self._impute_missing_sd(outcome)
            
            # RoB2偏倚风险评估
            rob2_assessment, tokens = await self._assess_rob2_bias_risk(
                abstract_chunks, methods_chunks, results_chunks  # 传入前5个chunks作为abstract_chunks
            )
            total_tokens = self._safe_add_tokens(total_tokens, tokens)
            
            # 提取TMS特定参数
            tms_metadata, tokens = await self._extract_tms_metadata_from_chunks(
                methods_chunks, table_chunks, study_design
            )
            total_tokens = self._safe_add_tokens(total_tokens, tokens)

            # 计算dropout_rate
            control_size = self._safe_get_int(basic_metadata, "control_sample_size")
            intervention_size = self._safe_get_int(basic_metadata, "intervention_sample_size")
            follow_up_people = self._safe_get_int(basic_metadata, "num_follow_up_people")
            sample_people = self._safe_get_int(basic_metadata, "sample_size")

            if (control_size is not None and intervention_size is not None and 
                follow_up_people is not None):
                total_sample = control_size + intervention_size
                if "baseline_characteristics" not in numerical_data:
                    numerical_data["baseline_characteristics"] = {}
                if total_sample != follow_up_people:
                    numerical_data["baseline_characteristics"]["dropout_rate"] = (
                        total_sample - follow_up_people
                    ) / total_sample
                else:
                    numerical_data["baseline_characteristics"]["dropout_rate"] = 0.0
            elif sample_people is not None and intervention_size is None and control_size is None and follow_up_people is not None:
                if "baseline_characteristics" not in numerical_data:
                    numerical_data["baseline_characteristics"] = {}
                if sample_people != follow_up_people:
                    numerical_data["baseline_characteristics"]["dropout_rate"] = (
                        sample_people - follow_up_people
                    ) / sample_people
                else:
                    numerical_data["baseline_characteristics"]["dropout_rate"] = 0.0
            else:
                logger.warning(f"无法计算dropout_rate: control_size={control_size}, "
                             f"intervention_size={intervention_size}, follow_up_people={follow_up_people}")

            # 合并所有元数据
            complete_metadata = {
                **basic_metadata,
                **study_design,
                **numerical_data,
                **tms_metadata,
                "outcome_definitions": outcome_definitions,
                "rob2_assessment": rob2_assessment,
                "paper_id": doi,
                "doi": doi,
                "extraction_timestamp": asyncio.get_event_loop().time(),
                "chunk_count": len(chunk_list),
                "has_tables": len(table_chunks) > 0
            }
            
            # 存储到evaluated_papers_storage
            await self.evaluated_papers_storage.upsert({doi: complete_metadata})

            title = complete_metadata.get('title') or 'Unknown'
            if isinstance(title, str):
                title_display = title[:50]
            else:
                title_display = 'Unknown'

            logger.info(f"论文元数据提取完成: {title_display}...")
            
            logger.info(f"论文元数据提取完成，总token消耗: {total_tokens}")
            return total_tokens
        
        except Exception as e:
            logger.error(f"论文元数据提取失败: {str(e)}")
            raise
        finally:
            await self._insert_done()

    def _find_chunks_by_type(self, chunks: list, section_type: str) -> list:
        """根据内容类型查找chunks"""
        section_keywords = {
            "abstract": ["abstract", "摘要", "summary","a b s t r a c t"],
            "methods": ["methods", "methodology", "方法", "participants",
                        "procedure","coil","intensity"],
            "results": ["results", "结果", "findings", "outcomes"],
            "discussion": ["discussion", "讨论", "conclusion", "结论"],
            "introduction": ["introduction", "背景", "background"]
        }
        
        keywords = section_keywords.get(section_type, [])
        matching_chunks = []
        
        for chunk in chunks:
            content_lower = chunk["content"].lower()
            if any(keyword in content_lower for keyword in keywords):
                matching_chunks.append(chunk)
        
        return matching_chunks

    def _find_table_chunks(self, chunks: list) -> list:
        """查找包含表格的chunks"""
        return [chunk for chunk in chunks if chunk.get("contains_table", False)]

    async def _extract_doi_from_content(self, paper_content: str) -> str:
        """从论文内容中提取DOI号，按优先级顺序查找"""

        # 更精确的DOI正则表达式模式
        doi_patterns = [
            r'DOI:\s*([10]\.\d+/[a-zA-Z0-9\.\-_\(\)/]+)',
            r'doi:\s*([10]\.\d+/[a-zA-Z0-9\.\-_\(\)/]+)',
            r'https?://doi\.org/([10]\.\d+/[a-zA-Z0-9\.\-_\(\)/]+)',
            r'https?://dx\.doi\.org/([10]\.\d+/[a-zA-Z0-9\.\-_\(\)/]+)',
            r'\b(10\.\d+/[a-zA-Z0-9\.\-_\(\)/]+)\b'
        ]
        
        # 优先级1: 检查MD文件第一行的HTML注释（来自页眉页脚提取）
        first_line = paper_content.split('\n')[0] if paper_content else ""
        comment_pattern = r'<!-- DOI: ([10]\.\d+/[a-zA-Z0-9\.\-_\(\)/]+) -->'
        comment_match = re.search(comment_pattern, first_line)
        if comment_match:
            doi = comment_match.group(1).rstrip('.,;:')
            # 清理DOI中的空格
            doi = self._clean_doi(doi)
            logger.info(f"从MD文件第一行HTML注释中提取到DOI: {doi}")
            return doi
        
        # 优先级2: 在文章开头寻找DOI（前20000字符）
        content_start = paper_content[:20000]
        for pattern in doi_patterns:
            match = re.search(pattern, content_start, re.IGNORECASE)
            if match:
                doi = match.group(1).rstrip('.,;:')
                # 清理DOI中的空格
                doi = self._clean_doi(doi)
                logger.info(f"在文章开头提取到DOI: {doi}")
                return doi
        
        # 优先级3: 在文章结尾寻找DOI（后20000字符）
        content_last = paper_content[-20000:]
        for pattern in doi_patterns:
            match = re.search(pattern, content_last, re.IGNORECASE)
            if match:
                doi = match.group(1).rstrip('.,;:')
                # 清理DOI中的空格
                doi = self._clean_doi(doi)
                logger.info(f"在文章结尾提取到DOI: {doi}")
                return doi
        
        # 优先级4: 使用LLM从开头提取
        try:
            doi = await self._extract_doi_with_llm(content_start)
            if doi:
                logger.info(f"通过LLM从开头提取到DOI: {doi}")
                return doi
        except Exception as e:
            logger.warning(f"LLM从开头提取DOI失败: {str(e)}")
        
        # 优先级5: 使用LLM从结尾提取
        try:
            doi = await self._extract_doi_with_llm(content_last)
            if doi:
                logger.info(f"通过LLM从结尾提取到DOI: {doi}")
                return doi
        except Exception as e:
            logger.warning(f"LLM从结尾提取DOI失败: {str(e)}")
        
        logger.warning("未找到DOI，将使用内容hash作为标识符")
        return None

    async def _extract_doi_with_llm(self, content_start: str) -> str:
        """使用LLM从论文内容中提取DOI"""
        
        doi_prompt = f"""
        从以下论文开头提取DOI号：
        
        {content_start}
        
        请只返回DOI号，格式如：10.1234/example.2023.123456
        如果没有找到DOI，请返回"None"。
        """
        
        try:
            response = await self.cheap_model_func(doi_prompt)
            response = response.strip()
            
            # 验证返回的是否是有效的DOI格式
            if re.match(r'^10\.\d+/', response):
                return response
            else:
                return None
        except Exception:
            return None

    def _clean_doi(self, doi: str) -> str:
        """清理DOI中的多余空格"""
        if not doi:
            return doi
        
        # 移除DOI中的所有空格
        cleaned_doi = re.sub(r'\s+', '', doi)
        
        # 移除末尾的标点符号，但保留DOI的有效字符
        cleaned_doi = re.sub(r'[^\w\.\-/\(\)]+$', '', cleaned_doi)
        
        # 验证清理后的DOI格式
        if re.match(r'^10\.\d+/', cleaned_doi):
            return cleaned_doi
        else:
            return doi  # 如果清理后格式不对，返回原始DOI

    async def _extract_basic_metadata_from_chunks(
        self, abstract_chunks: list, methods_chunks: list
    ) -> Tuple[dict, int]:
        """从特定类型的chunks中提取基础元数据，返回(元数据, token消耗)"""
        
        # 智能选择最相关的chunks，避免重复
        selected_chunks = []
        
        # 使用前2-3个chunks作为abstract_chunks（包含标题、作者、摘要等关键信息）
        if abstract_chunks:
            # 按chunk_order_index排序，确保顺序正确
            sorted_chunks = sorted(abstract_chunks, key=lambda x: x.get("chunk_order_index", 0))
            selected_chunks.extend(sorted_chunks[:3])  # 取前3个chunks
        
        # 选择包含关键词最多的methods chunk
        method_chunk = self._select_best_chunk_by_keywords(
            methods_chunks,
            ["design", "methods", "screen", "follow-up", "intervention", "procedure"]
        )
        if method_chunk:
            selected_chunks.append(method_chunk)

        sample_chunk = self._select_best_chunk_by_keywords(
            methods_chunks,
            ["participants", "randomize", "active", "sham", "n =", "n=", "withdrawn"]
        )
        if sample_chunk:
            selected_chunks.append(sample_chunk)
        
        # 合并选中的chunks
        combined_content = "\n\n".join([chunk["content"] for chunk in selected_chunks])

        metadata_prompt = PROMPTS["metadata_prompt"].format(combined_content=combined_content)
        
        try:
            response, tokens = await self.best_model_func(metadata_prompt, return_tokens=True)
            metadata = convert_response_to_json(response)
            tokens = int(tokens) if tokens else 0
            return self._validate_and_clean_metadata(metadata), tokens
        except Exception as e:
            logger.warning(f"基础元数据提取失败: {str(e)}")
            return self._get_default_metadata(), 0


    async def _extract_numerical_data_from_tables(
            self, table_chunks: list, outcome_definitions: dict, all_chunks: list = None,
            study_design: dict = None
    ) -> Tuple[dict, int]:
        """从表格chunks及其上下文中提取数值数据，返回(数据, token消耗)"""
        if not table_chunks or not outcome_definitions:
            return {}, 0

        primary_outcomes = outcome_definitions.get("primary_outcomes", [])
        secondary_outcomes = outcome_definitions.get("secondary_outcomes", [])

        if not primary_outcomes and not secondary_outcomes:
            logger.warning("没有找到结局指标定义，无法进行针对性数据提取")
            return {}, 0

        # 选择最有价值的表格chunks
        selected_tables = self._select_numerical_chunks(table_chunks)
        
        if not selected_tables:
            return {}, 0

        # 为每个表格找到上下文chunks
        table_with_context = []
        for table_chunk in selected_tables[:2]:
            context_chunks = self._find_table_context_chunks(table_chunk, all_chunks or table_chunks)
            table_with_context.append({
                'table': table_chunk,
                'context': context_chunks
            })

        # 构建目标指标列表
        primary_target_scales = []
        secondary_target_scales = []
        for outcome in primary_outcomes:
            if outcome.get("scale"):
                primary_target_scales.append(outcome["scale"])
            if outcome.get("name"):
                primary_target_scales.append(outcome["name"])

        for outcome in secondary_outcomes:
            if outcome.get("scale"):
                secondary_target_scales.append(outcome["scale"])

        # 构建包含表格和上下文的内容
        combined_content = ""
        for item in table_with_context:
            # 添加表格前的上下文
            for ctx in item['context']['before']:
                combined_content += f"[上下文] {ctx['content']}\n\n"
            
            # 添加表格内容
            combined_content += f"[表格] {item['table']['content']}\n\n"
            
            # 添加表格后的上下文
            for ctx in item['context']['after']:
                combined_content += f"[上下文] {ctx['content']}\n\n"
            
            # # 控制总长度
            # if len(combined_content) > 60000:
            #     break

        design_type = study_design.get("study_design_type") if study_design else "two-arm rct"

        if design_type == "double-blind sham-controlled crossover":
            prompt_template = PROMPTS["numerical_prompt_crossover"]
        elif design_type == "multi-arm rct":
            prompt_template = PROMPTS["numerical_prompt_multi_arm"]
        else:  # two-arm rct, open label, case report
            prompt_template = PROMPTS["numerical_prompt"]

        numerical_prompt = prompt_template.format(
            table_content=combined_content,
            primary_outcomes=primary_target_scales,
            secondary_outcomes=secondary_target_scales
        )
        
        try:
            response, tokens = await self.best_model_func(numerical_prompt, return_tokens=True)
            numerical_data = convert_response_to_json(response)
            
            # 确保返回整数
            tokens = int(tokens) if tokens else 0
            
            return numerical_data, tokens
        except Exception as e:
            logger.warning(f"数值数据提取失败: {str(e)}")
            return {}, 0

    def _find_table_context_chunks(self, table_chunk: dict, all_chunks: list) -> dict:
        """为表格chunk找到相关的上下文chunks"""
        table_order = table_chunk.get("chunk_order_index", 0)
        table_doc_id = table_chunk.get("full_doc_id", "")
        
        # 找到同一文档中的相邻chunks
        same_doc_chunks = [
            chunk for chunk in all_chunks 
            if chunk.get("full_doc_id") == table_doc_id and 
            not chunk.get("contains_table", False)  # 排除其他表格chunks
        ]
        
        # 按chunk_order_index排序
        same_doc_chunks.sort(key=lambda x: x.get("chunk_order_index", 0))
        
        before_chunks = []
        after_chunks = []
        
        # 查找表格前后的chunks（优先选择results相关的）
        for chunk in same_doc_chunks:
            chunk_order = chunk.get("chunk_order_index", 0)
            chunk_content = chunk["content"].lower()
            
            # 检查是否包含results相关关键词
            results_keywords = [
                "result", "outcome", "table",
                "data", "performance", "significant",
                "mean", "sd", "p =", "summarize",
                "note", "abbreviation"
            ]
            
            is_results_related = any(keyword in chunk_content for keyword in results_keywords)
            
            if chunk_order < table_order and len(before_chunks) < 2:
                # 优先选择results相关的chunk(可扩大范围)，否则就是距离最近的chunk
                if is_results_related and abs(chunk_order - table_order) <= 2:
                    before_chunks.append(chunk)
                elif abs(chunk_order - table_order) <= 1:
                    before_chunks.append(chunk)
            elif chunk_order > table_order and len(after_chunks) < 2:
                # 优先选择results相关的chunk(可扩大范围)，否则就是距离最近的chunk
                if is_results_related and abs(chunk_order - table_order) <= 2:
                    after_chunks.append(chunk)
                elif abs(chunk_order - table_order) <= 1:
                    after_chunks.append(chunk)
        
        # 按距离表格的远近排序
        before_chunks.sort(key=lambda x: table_order - x.get("chunk_order_index", 0))
        after_chunks.sort(key=lambda x: x.get("chunk_order_index", 0) - table_order)
        
        return {
            'before': before_chunks[:2],  # 最多取前2个
            'after': after_chunks[:2]     # 最多取后2个
        }

    async def _identify_study_design(self, abstract_chunks: list, methods_chunks: list) -> Tuple[dict, int]:
        """识别研究设计类型"""

        relevant_chunks = abstract_chunks[:2] + methods_chunks[:2]
        combined_content = "\n\n".join([chunk["content"] for chunk in relevant_chunks])

        design_prompt = PROMPTS["study_design_identification"].format(
            combined_content=combined_content
        )

        try:
            response, tokens = await self.best_model_func(design_prompt, return_tokens=True)
            study_design = convert_response_to_json(response)

            # 验证和设置默认值
            if not study_design.get("study_design_type"):
                study_design["study_design_type"] = "two-arm rct"  # 默认为双臂

            tokens = int(tokens) if tokens else 0

            return study_design, tokens

        except Exception as e:
            logger.warning(f"研究设计识别失败: {str(e)}")
            return {
                "study_design_type": "two_arm_rct",
                "design_details": {}
            }, 0


    async def _extract_outcome_definitions(self, methods_chunks: list) -> Tuple[dict, int]:
        """从文本中提取结局指标定义，返回(定义, token消耗)"""

        # 扩大搜索范围，不仅限于包含明确"结局指标"关键词的chunks
        relevant_chunks = []

        # 第一优先级：包含明确结局指标关键词的chunks
        explicit_outcome_keywords = [
            "primary outcome", "secondary outcome", "primary endpoint", "secondary endpoint",
            "main outcome", "primary measure", "secondary measure", "outcome measure",
            "主要结局", "次要结局", "主要终点", "次要终点", "结局指标"
        ]

        # 第二优先级：包含量表/评估工具关键词的chunks
        scale_keywords = [
            "scale", "assessment", "inventory", "questionnaire", "test", "battery",
            "mmse", "moca", "adas-cog", "hamd", "madrs", "bdi", "hama", "gad",
            "量表", "评估", "测试", "问卷", "评定", "检查"
        ]

        # 第三优先级：包含测量/评价相关关键词的chunks
        measurement_keywords = [
            "measured", "assessed", "evaluated", "administered", "scored",
            "cognitive", "depression", "anxiety", "function", "performance",
            "测量", "评价", "评估", "施测", "计分", "认知", "抑郁", "焦虑", "功能"
        ]

        # 按优先级收集相关chunks
        priority_chunks = []
        for chunk in methods_chunks:
            content_lower = chunk["content"].lower()

            # 检查是否包含明确的结局指标关键词
            if any(keyword in content_lower for keyword in explicit_outcome_keywords):
                priority_chunks.append({"chunk": chunk, "priority": 1})
            # 检查是否包含量表关键词
            elif any(keyword in content_lower for keyword in scale_keywords):
                priority_chunks.append({"chunk": chunk, "priority": 2})
            # 检查是否包含测量关键词
            elif any(keyword in content_lower for keyword in measurement_keywords):
                priority_chunks.append({"chunk": chunk, "priority": 3})

        if not priority_chunks:
            logger.warning("未找到包含结局指标或量表信息的chunks")
            return {"primary_outcomes": [], "secondary_outcomes": []}, 0

        # 按优先级排序，优先使用高优先级的chunks
        priority_chunks.sort(key=lambda x: x["priority"])

        # 选择最多3个最相关的chunks
        relevant_chunks = [item["chunk"] for item in priority_chunks[:3]]

        # 合并相关内容
        combined_content = "\n\n".join([chunk["content"] for chunk in relevant_chunks])

        outcome_prompt = PROMPTS["outcome_prompt"].format(combined_content=combined_content)

        try:
            response, tokens = await self.best_model_func(outcome_prompt, return_tokens=True)
            outcome_definitions = convert_response_to_json(response)
            validated_outcomes = self._validate_outcome_definitions(outcome_definitions)
            
            # 确保返回整数
            tokens = int(tokens) if tokens else 0
            
            logger.info(f"提取到 {len(validated_outcomes['primary_outcomes'])} 个主要结局指标, "
                        f"{len(validated_outcomes['secondary_outcomes'])} 个次要结局指标")
            
            return validated_outcomes, tokens

        except Exception as e:
            logger.warning(f"结局指标定义提取失败: {str(e)}")
            return {"primary_outcomes": [], "secondary_outcomes": []}, 0

    async def _extract_tms_metadata_from_chunks(
        self, methods_chunks: list, table_chunks: list, study_design: dict = None
    ) -> Tuple[dict, int]:
        """从chunks中提取TMS特定元数据"""
        
        # 合并所有chunks并检查是否TMS相关
        all_chunks = methods_chunks + table_chunks
        if not all_chunks:
            return {}, 0

        design_type = study_design.get("study_design_type") if study_design else "two-arm rct"

        if design_type == "multi-arm rct":
            # 多臂RCT：提取每个干预组的TMS参数
            return await self._extract_tms_metadata_for_multi_arm(all_chunks)
        else:
            # 双臂RCT或其他设计：提取单一TMS参数
            return await self._extract_tms_metadata_single_arm(all_chunks)

    async def _extract_tms_metadata_single_arm(self, all_chunks: list) -> Tuple[dict, int]:
        """提取单臂或双臂RCT的TMS元数据"""
        # 选择TMS相关性最高的chunk
        tms_chunk = self._select_best_chunk_by_keywords(
            all_chunks,
            ["tms", "rtms", "frequency", "intensity", "coil", "intervention",
             "train duration", "rmt", "session", "hz"]
        )

        if not tms_chunk:
            return {}, 0

        tms_prompt = PROMPTS["tms_prompt"].format(content=tms_chunk["content"])

        try:
            response, tokens = await self.best_model_func(tms_prompt, return_tokens=True)
            tms_data = convert_response_to_json(response)
            tokens = int(tokens) if tokens else 0
            return tms_data, tokens
        except Exception as e:
            logger.warning(f"TMS元数据提取失败: {str(e)}")
            return {}, 0

    async def _extract_tms_metadata_for_multi_arm(self, all_chunks: list) -> Tuple[dict, int]:
        """提取多臂RCT的TMS元数据（每个干预组可能有不同的刺激参数）"""

        # 选择所有TMS相关的chunks（可能需要多个chunks来覆盖所有组）
        tms_chunks = []
        for chunk in all_chunks:
            content_lower = chunk["content"].lower()
            tms_keywords = ["tms", "rtms", "frequency", "intensity", "coil",
                            "intervention", "train duration", "rmt", "session", "hz",
                            "group", "arm"]
            if any(keyword in content_lower for keyword in tms_keywords):
                tms_chunks.append(chunk)

        if not tms_chunks:
            return {}, 0

        # 合并相关chunks的内容
        combined_content = "\n\n".join([chunk["content"] for chunk in tms_chunks[:5]])

        tms_multi_arm_prompt = PROMPTS["tms_prompt_multi_arm"].format(
            content=combined_content
        )

        try:
            response, tokens = await self.best_model_func(tms_multi_arm_prompt, return_tokens=True)
            tms_data = convert_response_to_json(response)
            tokens = int(tokens) if tokens else 0
            return tms_data, tokens

        except Exception as e:
            logger.warning(f"多臂RCT的TMS元数据提取失败: {str(e)}")
            return {}, 0


    def _select_best_chunk_by_keywords(self, chunks: list, keywords: list) -> dict:
        """根据关键词选择最相关的chunk"""
        if not chunks:
            return None
        
        best_chunk = None
        max_score = 0
        
        for chunk in chunks:
            content_lower = chunk["content"].lower()
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > max_score:
                max_score = score
                best_chunk = chunk
        
        return best_chunk if max_score > 0 else chunks[0]

    def _select_numerical_chunks(self, table_chunks: list) -> list:
        """选择包含数值数据的表格chunks"""
        numerical_keywords = [
            "mean", "sd", "p-value",  "effect size",
            "baseline", "follow-up", "±", "95%", "active", "sham", "t/z"
        ]
        
        scored_chunks = []
        for chunk in table_chunks:
            content_lower = chunk["content"].lower()
            score = sum(1 for keyword in numerical_keywords if keyword in content_lower)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # 按分数排序，返回最相关的chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks]


    def _validate_and_clean_metadata(self, metadata: dict) -> dict:
        """验证和清理元数据"""
        if not isinstance(metadata, dict):
            return self._get_default_metadata()

        # 确保必要字段存在
        required_fields = ["title", "study_type", "sample_size"]
        for field in required_fields:
            if field not in metadata:
                metadata[field] = None

        # 数据类型验证
        if metadata.get("year") and not isinstance(metadata["year"], int):
            try:
                metadata["year"] = int(metadata["year"])
            except (ValueError, TypeError):
                metadata["year"] = None
        
        # 如果没有年份，尝试从DOI中提取
        if not metadata.get("year") and metadata.get("doi"):
            extracted_year = self._extract_year_from_doi(metadata["doi"])
            if extracted_year:
                metadata["year"] = extracted_year
                logger.info(f"从DOI中提取到年份: {extracted_year}")

        if metadata.get("sample_size") and not isinstance(metadata["sample_size"], int):
            try:
                metadata["sample_size"] = int(metadata["sample_size"])
            except (ValueError, TypeError):
                metadata["sample_size"] = None

        return metadata

    def _get_default_metadata(self) -> dict:
        """获取默认元数据结构"""
        return {
            "title": None,
            "authors": [],
            "journal": None,
            "year": None,
            "study_type": None,
            "sample_size": None,
            "intervention": None,
            "control": None,
            "outcome_measures": [],
            "bias_risk": None,
            "quality_score": None,
            "inclusion_criteria": None,
            "exclusion_criteria": None,
            "blinding": None,
            "randomization": None
        }

    def _validate_outcome_definitions(self, outcome_definitions: dict) -> dict:
        """验证和清理结局指标定义"""
        if not isinstance(outcome_definitions, dict):
            return {"primary_outcomes": [], "secondary_outcomes": []}

        validated = {"primary_outcomes": [], "secondary_outcomes": []}

        # 验证主要结局指标
        for outcome in outcome_definitions.get("primary_outcomes", []):
            if isinstance(outcome, dict) and outcome.get("name"):
                validated["primary_outcomes"].append(outcome)

        # 验证次要结局指标
        for outcome in outcome_definitions.get("secondary_outcomes", []):
            if isinstance(outcome, dict) and outcome.get("name"):
                validated["secondary_outcomes"].append(outcome)

        return validated

    async def _assess_rob2_bias_risk(self, abstract_chunks: list, methods_chunks: list, results_chunks: list) -> Tuple[dict, int]:
        """使用RoB2工具评估偏倚风险"""
        
        # 选择包含方法学信息的chunks
        relevant_chunks = []
        
        # 方法学关键词
        methodology_keywords = [
            "randomization", "randomisation", "blinding", "allocation", "concealment",
            "intention-to-treat", "per-protocol", "dropout", "withdrawal", "missing data",
            "method", "participant", "procedure",
            "随机", "盲法", "分配", "隐藏", "意向性治疗", "脱落"
        ]
        
        # 使用set来避免重复chunks，基于chunk的唯一标识符
        seen_chunk_ids = set()
        all_chunks = []
        acknowledgments_chunk = None
        
        # 按优先级添加chunks：methods > abstract > results
        for chunk_list in [methods_chunks, abstract_chunks, results_chunks]:
            for chunk in chunk_list:
                chunk_id = chunk.get("chunk_order_index", id(chunk))  # 使用chunk的唯一标识
                if chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

        # 从去重后的chunks中筛选包含方法学关键词的chunks
        for chunk in all_chunks:
            content_lower = chunk["content"].lower()
            if any(keyword in content_lower for keyword in methodology_keywords):
                relevant_chunks.append(chunk)
            if "acknowledgments" in content_lower:
                acknowledgments_chunk = chunk
        
        if not relevant_chunks:
            logger.warning("未找到包含方法学信息的chunks")
            return self._get_default_rob2_assessment(), 0
        
        # 合并相关内容
        combined_content = "\n\n".join([chunk["content"] for chunk in relevant_chunks[:3]])
        # 添加致谢chunk
        if acknowledgments_chunk:
            combined_content += f"\n\n{acknowledgments_chunk['content']}"
        
        rob2_prompt = PROMPTS["rob2_prompt"].format(combined_content=combined_content)
        
        try:
            response, tokens = await self.best_model_func(rob2_prompt, return_tokens=True)
            rob2_assessment = convert_response_to_json(response)
            
            # 验证和清理RoB2评估结果
            validated_assessment = self._validate_rob2_assessment(rob2_assessment)

            tokens = int(tokens) if tokens else 0
            
            logger.info(f"RoB2评估完成，总体偏倚风险: {validated_assessment.get('overall_bias_risk', {}).get('risk_level', 'Unknown')}")
            
            return validated_assessment, tokens
            
        except Exception as e:
            logger.warning(f"RoB2偏倚风险评估失败: {str(e)}")
            return self._get_default_rob2_assessment(), 0

    def _validate_rob2_assessment(self, assessment: dict) -> dict:
        """验证和清理RoB2评估结果"""
        if not isinstance(assessment, dict):
            return self._get_default_rob2_assessment()
        
        valid_risk_levels = ["Low", "Some concerns", "High"]
        domains = [
            "domain1_randomization",
            "domain2_deviations", 
            "domain3_missing_data",
            "domain4_outcome_measurement",
            "domain5_selective_reporting",
            "overall_bias_risk"
        ]
        
        validated = {}
        
        for domain in domains:
            domain_data = assessment.get(domain, {})
            if isinstance(domain_data, dict):
                risk_level = domain_data.get("risk_level", "Some concerns")
                if risk_level not in valid_risk_levels:
                    risk_level = "Some concerns"
                
                validated[domain] = {
                    "risk_level": risk_level,
                    "rationale": domain_data.get("rationale", "信息不足"),
                    "supporting_evidence": domain_data.get("supporting_evidence", "")
                }
            else:
                validated[domain] = {
                    "risk_level": "Some concerns",
                    "rationale": "信息不足",
                    "supporting_evidence": ""
                }
        
        return validated

    def _get_default_rob2_assessment(self) -> dict:
        """获取默认的RoB2评估结果"""
        default_domain = {
            "risk_level": "Some concerns",
            "rationale": "信息不足，无法准确评估",
            "supporting_evidence": ""
        }
        
        return {
            "domain1_randomization": default_domain.copy(),
            "domain2_deviations": default_domain.copy(),
            "domain3_missing_data": default_domain.copy(), 
            "domain4_outcome_measurement": default_domain.copy(),
            "domain5_selective_reporting": default_domain.copy(),
            "overall_bias_risk": default_domain.copy()
        }

    def _is_null_value(self, value) -> bool:
        """检查值是否为空值（包括None、"null"、空字符串等）"""
        return value is None or value == "null" or value == "" or value == "None"

    def _safe_get_numeric(self, data: dict, key: str, default=None):
        """安全获取数值，处理null字符串"""
        value = data.get(key, default)
        if self._is_null_value(value):
            return None
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _safe_get_int(self, data: dict, key: str, default=None):
        """安全获取整数，处理null字符串"""
        value = data.get(key, default)
        if self._is_null_value(value):
            return None
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _safe_add_tokens(self, current_total: int, new_tokens) -> int:
        """安全地累加token数量"""
        if new_tokens is None:
            return current_total
        
        if isinstance(new_tokens, str):
            try:
                return current_total + int(new_tokens)
            except (ValueError, TypeError):
                logger.warning(f"无法转换token值: {new_tokens}")
                return current_total
        
        if isinstance(new_tokens, (int, float)):
            return current_total + int(new_tokens)
        
        logger.warning(f"未知的token类型: {type(new_tokens)}")
        return current_total

    def _calculate_sd_from_ci(self, mean: float, ci_lower: float, ci_upper: float, 
                             ci_percent: float, n: int) -> float:
        """从置信区间逆推标准差"""
        try:
            # 检查输入参数是否有效
            if any(self._is_null_value(x) for x in [mean, ci_lower, ci_upper, n]):
                logger.warning(f"CI逆推SD参数无效: mean={mean}, ci_lower={ci_lower}, ci_upper={ci_upper}, n={n}")
                return None
            
            # 转换为数值类型
            mean = float(mean)
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
            n = int(n)
            ci_percent = float(ci_percent) if not self._is_null_value(ci_percent) else 95
            
            # 计算置信区间的半宽
            ci_half_width = (ci_upper - ci_lower) / 2
            
            # 根据置信区间百分比确定t值或z值
            if ci_percent == 95:
                if n >= 30:
                    critical_value = 1.96  # z值
                else:
                    from scipy import stats
                    critical_value = stats.t.ppf(0.975, n-1)  # t值
            elif ci_percent == 90:
                if n >= 30:
                    critical_value = 1.645
                else:
                    from scipy import stats
                    critical_value = stats.t.ppf(0.95, n-1)
            elif ci_percent == 99:
                if n >= 30:
                    critical_value = 2.576
                else:
                    from scipy import stats
                    critical_value = stats.t.ppf(0.995, n-1)
            else:
                # 默认使用95%置信区间
                critical_value = 1.96
            
            # SE = CI半宽 / 临界值
            se = ci_half_width / critical_value
            
            # SD = SE * sqrt(n)
            sd = se * (n ** 0.5)
            
            logger.info(f"从CI逆推SD: mean={mean}, CI=[{ci_lower}, {ci_upper}], n={n}, SD={sd:.3f}")
            return sd
            
        except Exception as e:
            logger.warning(f"从CI逆推SD失败: {str(e)}")
            return None

    def _calculate_sd_from_mean_difference(self, mean_diff: float, diff_ci_lower: float,
                                         diff_ci_upper: float, n1: int, n2: int) -> tuple:
        """从均值差及其CI逆推两组的合并标准差"""
        try:
            # 计算差值CI的半宽
            diff_ci_half_width = (diff_ci_upper - diff_ci_lower) / 2
            
            # 使用95%置信区间的z值（通常报告的是95% CI）
            z_value = 1.96
            
            # SE_diff = CI半宽 / z值
            se_diff = diff_ci_half_width / z_value
            
            # 对于两独立样本t检验：SE_diff = sqrt(SD1²/n1 + SD2²/n2)
            # 假设两组方差相等：SD1 = SD2 = SD_pooled
            # 则：SE_diff = SD_pooled * sqrt(1/n1 + 1/n2)
            
            pooled_sd = se_diff / ((1/n1 + 1/n2) ** 0.5)
            
            logger.info(f"从均值差CI逆推合并SD: diff={mean_diff}, CI=[{diff_ci_lower}, {diff_ci_upper}], SD_pooled={pooled_sd:.3f}")
            return pooled_sd, pooled_sd
            
        except Exception as e:
            logger.warning(f"从均值差CI逆推SD失败: {str(e)}")
            return None, None

    def _calculate_sd_from_p_value(self, mean1: float, mean2: float, n1: int, n2: int, 
                                  p_value: float = None, t_value: float = None) -> tuple:
        """从p值或t值逆推标准差"""
        try:
            from scipy import stats
            import numpy as np
            
            mean_diff = abs(mean1 - mean2)
            df = n1 + n2 - 2  # 自由度
            
            if t_value is not None:
                # 直接使用t值
                t_stat = abs(t_value)
            elif p_value is not None:
                # 从p值逆推t值（假设是双尾检验）
                t_stat = abs(stats.t.ppf(p_value/2, df))
            else:
                logger.warning("需要提供p值或t值")
                return None, None
            
            # 对于两独立样本t检验：
            # t = (mean1 - mean2) / (SD_pooled * sqrt(1/n1 + 1/n2))
            # 因此：SD_pooled = (mean1 - mean2) / (t * sqrt(1/n1 + 1/n2))
            
            pooled_sd = mean_diff / (t_stat * ((1/n1 + 1/n2) ** 0.5))
            
            logger.info(f"从统计量逆推SD: mean_diff={mean_diff}, t={t_stat}, p={p_value}, SD_pooled={pooled_sd:.3f}")
            return pooled_sd, pooled_sd
            
        except Exception as e:
            logger.warning(f"从统计量逆推SD失败: {str(e)}")
            return None, None

    def _impute_missing_sd(self, outcome_data: dict) -> dict:
        """补充缺失的标准差"""
        if not outcome_data:
            return outcome_data

        intervention = outcome_data.get("intervention_group") or {}
        control = outcome_data.get("control_group") or {}
        effect_size_data = outcome_data.get("effect_size") or {}
        
        # 检查是否需要补充SD（包括"null"字符串）
        need_intervention_sd = self._is_null_value(intervention.get("sd", "null"))
        need_control_sd = self._is_null_value(control.get("sd", "null"))
        
        if not (need_intervention_sd or need_control_sd):
            return outcome_data  # 不需要补充
        
        logger.info(f"尝试补充缺失的SD: 干预组={need_intervention_sd}, 对照组={need_control_sd}")
        
        # 方法1: 从置信区间逆推
        if need_intervention_sd:
            mean = self._safe_get_numeric(intervention, "mean")
            ci_lower = self._safe_get_numeric(intervention, "ci_lower")
            ci_upper = self._safe_get_numeric(intervention, "ci_upper")
            n = self._safe_get_int(intervention, "n")
            ci_percent = self._safe_get_numeric(intervention, "ci_percent") or 95
            
            if all(x is not None for x in [mean, ci_lower, ci_upper, n]):
                sd = self._calculate_sd_from_ci(mean, ci_lower, ci_upper, ci_percent, n)
                if sd:
                    intervention["sd"] = round(sd, 3)
                    intervention["sd_method"] = "calculated_from_ci"
                    need_intervention_sd = False  # 已成功计算
        
        if need_control_sd:
            mean = self._safe_get_numeric(control, "mean")
            ci_lower = self._safe_get_numeric(control, "ci_lower")
            ci_upper = self._safe_get_numeric(control, "ci_upper")
            n = self._safe_get_int(control, "n")
            ci_percent = self._safe_get_numeric(control, "ci_percent") or 95
            
            if all(x is not None for x in [mean, ci_lower, ci_upper, n]):
                sd = self._calculate_sd_from_ci(mean, ci_lower, ci_upper, ci_percent, n)
                if sd:
                    control["sd"] = round(sd, 3)
                    control["sd_method"] = "calculated_from_ci"
                    need_control_sd = False  # 已成功计算
        
        # 如果都已经计算出来了，直接返回
        if not (need_intervention_sd or need_control_sd):
            outcome_data["intervention_group"] = intervention
            outcome_data["control_group"] = control
            return outcome_data
        
        # 方法2: 从均值差的CI逆推（如果两组都还缺少SD）
        if need_intervention_sd and need_control_sd:
            # 获取效应量数据
            effect_ci_lower = self._safe_get_numeric(effect_size_data, "ci_lower")
            effect_ci_upper = self._safe_get_numeric(effect_size_data, "ci_upper")
            
            # 获取干预组数据
            int_mean = self._safe_get_numeric(intervention, "mean")
            int_n = self._safe_get_int(intervention, "n")
            
            # 获取对照组数据
            ctrl_mean = self._safe_get_numeric(control, "mean")
            ctrl_n = self._safe_get_int(control, "n")
            
            if all(x is not None for x in [effect_ci_lower, effect_ci_upper, int_mean, int_n, ctrl_mean, ctrl_n]):
                mean_diff = abs(int_mean - ctrl_mean)
                sd1, sd2 = self._calculate_sd_from_mean_difference(
                    mean_diff, effect_ci_lower, effect_ci_upper, int_n, ctrl_n
                )
                
                if sd1 and sd2:
                    if need_intervention_sd:
                        intervention["sd"] = round(sd1, 3)
                        intervention["sd_method"] = "calculated_from_mean_diff_ci"
                        need_intervention_sd = False
                    if need_control_sd:
                        control["sd"] = round(sd2, 3)
                        control["sd_method"] = "calculated_from_mean_diff_ci"
                        need_control_sd = False
        
        # 如果都已经计算出来了，直接返回
        if not (need_intervention_sd or need_control_sd):
            outcome_data["intervention_group"] = intervention
            outcome_data["control_group"] = control
            return outcome_data
        
        # 方法3: 从p值或t值逆推
        if need_intervention_sd and need_control_sd:
            int_mean = self._safe_get_numeric(intervention, "mean")
            int_n = self._safe_get_int(intervention, "n")
            ctrl_mean = self._safe_get_numeric(control, "mean")
            ctrl_n = self._safe_get_int(control, "n")
            
            p_value = self._safe_get_numeric(effect_size_data, "p_value")
            t_value = self._safe_get_numeric(effect_size_data, "t_value")
            
            if (all(x is not None for x in [int_mean, int_n, ctrl_mean, ctrl_n]) and 
                (p_value is not None or t_value is not None)):
                
                sd1, sd2 = self._calculate_sd_from_p_value(
                    int_mean, ctrl_mean, int_n, ctrl_n, p_value, t_value
                )
                
                if sd1 and sd2:
                    if need_intervention_sd:
                        intervention["sd"] = round(sd1, 3)
                        intervention["sd_method"] = "calculated_from_statistics"
                    if need_control_sd:
                        control["sd"] = round(sd2, 3)
                        control["sd_method"] = "calculated_from_statistics"
        
        # 更新outcome_data
        outcome_data["intervention_group"] = intervention
        outcome_data["control_group"] = control
        
        return outcome_data

    def _impute_missing_sd_for_multi_arm(self, numerical_data: dict) -> dict:
        """补充多臂RCT缺失的标准差"""
        if not numerical_data:
            return numerical_data

        # 处理主要结局指标
        if numerical_data.get("primary_outcomes"):
            for outcome in numerical_data["primary_outcomes"]:
                if outcome and outcome.get("groups"):
                    for group in outcome["groups"]:
                        if self._is_null_value(group.get("sd")):
                            sd = self._calculate_sd_from_group_data(group)
                            if sd:
                                group["sd"] = round(sd, 3)
                                group["sd_method"] = "calculated_from_ci"

        # 处理次要结局指标
        if numerical_data.get("secondary_outcomes"):
            for outcome in numerical_data["secondary_outcomes"]:
                if outcome and outcome.get("groups"):
                    for group in outcome["groups"]:
                        if self._is_null_value(group.get("sd")):
                            sd = self._calculate_sd_from_group_data(group)
                            if sd:
                                group["sd"] = round(sd, 3)
                                group["sd_method"] = "calculated_from_ci"

        return numerical_data

    def _impute_missing_sd_for_crossover(self, numerical_data: dict) -> dict:
        """补充交叉设计缺失的标准差"""
        if not numerical_data:
            return numerical_data

        # 处理主要结局指标
        if numerical_data.get("primary_outcomes"):
            for outcome in numerical_data["primary_outcomes"]:
                if outcome and outcome.get("periods"):
                    for period in outcome["periods"]:
                        if period.get("conditions"):
                            for condition in period["conditions"]:
                                if self._is_null_value(condition.get("sd")):
                                    sd = self._calculate_sd_from_group_data(condition)
                                    if sd:
                                        condition["sd"] = round(sd, 3)
                                        condition["sd_method"] = "calculated_from_ci"

        # 处理次要结局指标
        if numerical_data.get("secondary_outcomes"):
            for outcome in numerical_data["secondary_outcomes"]:
                if outcome and outcome.get("periods"):
                    for period in outcome["periods"]:
                        if period.get("conditions"):
                            for condition in period["conditions"]:
                                if self._is_null_value(condition.get("sd")):
                                    sd = self._calculate_sd_from_group_data(condition)
                                    if sd:
                                        condition["sd"] = round(sd, 3)
                                        condition["sd_method"] = "calculated_from_ci"

        return numerical_data

    def _calculate_sd_from_group_data(self, group_data: dict) -> float:
        """从单个组的数据中计算SD（通用方法）"""
        if not group_data:
            return None

        # 尝试从CI计算SD
        mean = self._safe_get_numeric(group_data, "mean")
        ci_lower = self._safe_get_numeric(group_data, "ci_lower")
        ci_upper = self._safe_get_numeric(group_data, "ci_upper")
        n = self._safe_get_int(group_data, "n")
        ci_percent = self._safe_get_numeric(group_data, "ci_percent") or 95

        if all(x is not None for x in [mean, ci_lower, ci_upper, n]):
            return self._calculate_sd_from_ci(mean, ci_lower, ci_upper, ci_percent, n)

        return None

    def _extract_year_from_doi(self, doi: str) -> Optional[int]:
        """从DOI中提取年份"""
        if not doi:
            return None
        
        # DOI中常见的年份模式
        year_patterns = [
            r'\.(\d{4})\.',  # 如 10.1016/j.jad.2023.01.001
            r'/(\d{4})/',    # 如 10.1038/s41598-2023-12345-6
            r'\.(\d{4})$',   # 如 10.1001/jama.2023
            r'-(\d{4})-',    # 如 10.1038/s41598-2023-12345-6
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, doi)
            if matches:
                # 取最后一个匹配的年份（通常是发表年份）
                year = int(matches[-1])
                # 验证年份合理性（1900-2030）
                if 1900 <= year <= 2030:
                    return year
        
        return None

    async def _cleanup_failed_paper(self, inserting_chunks: dict) -> None:
        """清理提取失败的论文数据"""
        try:
            # 获取论文ID
            if inserting_chunks:
                first_chunk = next(iter(inserting_chunks.values()))
                paper_id = first_chunk.get("full_doc_id")
                
                if paper_id:
                    logger.info(f"清理失败论文的数据: {paper_id}")
                    
                    # 删除分块数据
                    chunk_keys = list(inserting_chunks.keys())
                    for chunk_key in chunk_keys:
                        await self.raw_papers_chunks.delete(chunk_key)
                    
                    # 删除原始论文数据
                    await self.raw_papers_storage.delete(paper_id)
                    
                    logger.info(f"已清理论文 {paper_id} 的所有数据")
        except Exception as cleanup_error:
            logger.error(f"清理失败论文数据时出错: {str(cleanup_error)}")

    def print_statistics(self):
        """打印处理统计信息"""
        
        if self._stats["total_papers_processed"] == 0:
            logger.info("尚未处理任何论文")
            return
        
        avg_time = self._stats["total_time_seconds"] / self._stats["total_papers_processed"]
        avg_tokens = self._stats["total_tokens_used"] / self._stats["total_papers_processed"]
        
        print("\n" + "="*60)
        print("Meta分析论文处理统计")
        print("="*60)
        print(f"总处理论文数: {self._stats['total_papers_processed']}")
        print(f"总处理时间: {self._stats['total_time_seconds']:.2f} 秒")
        print(f"总Token消耗: {self._stats['total_tokens_used']:,}")
        print(f"\n平均每篇论文:")
        print(f"  - 处理时间: {avg_time:.2f} 秒")
        print(f"  - Token消耗: {avg_tokens:,.0f}")
        print(f"\n处理速度: {3600/avg_time:.1f} 篇/小时")
        print("="*60 + "\n")

    def get_statistics(self) -> dict:
        """获取统计信息字典"""
        
        if self._stats["total_papers_processed"] == 0:
            return self._stats
        
        return {
            **self._stats,
            "avg_time_per_paper": self._stats["total_time_seconds"] / self._stats["total_papers_processed"],
            "avg_tokens_per_paper": self._stats["total_tokens_used"] / self._stats["total_papers_processed"],
            "papers_per_hour": 3600 / (self._stats["total_time_seconds"] / self._stats["total_papers_processed"])
        }

    def _write_paper_statistics(self, paper_info: dict):
        """将单篇论文的统计信息写入txt文件"""
        stats_file = os.path.join(self.working_dir, "paper_processing_stats.txt")

        # 计算平均值
        avg_time = self._stats["total_time_seconds"] / self._stats["total_papers_processed"]
        avg_tokens = self._stats["total_tokens_used"] / self._stats["total_papers_processed"]

        # 追加模式写入
        with open(stats_file, 'a', encoding='utf-8') as f:
            # 如果是第一篇论文（且文件为空或不存在），写入表头
            if self._stats["total_papers_processed"] == 1 or os.path.getsize(stats_file) == 0:
                f.write("=" * 80 + "\n")
                f.write("论文处理统计记录\n")
                f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

            # 写入单篇论文信息
            f.write(f"论文 #{self._stats['total_papers_processed']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"论文ID: {paper_info.get('paper_id', 'Unknown')}\n")
            f.write(f"标题: {paper_info.get('title', 'Unknown')[:100]}...\n")
            f.write(f"处理时间: {paper_info['processing_time']:.2f} 秒\n")
            f.write(f"Token消耗: {paper_info['tokens_used']:,}\n")
            f.write(f"处理时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            # 写入累计统计
            f.write(f"【累计统计】\n")
            f.write(f"已处理论文数: {self._stats['total_papers_processed']}\n")
            f.write(f"累计处理时间: {self._stats['total_time_seconds']:.2f} 秒\n")
            f.write(f"累计Token消耗: {self._stats['total_tokens_used']:,}\n")
            f.write(f"平均处理时间: {avg_time:.2f} 秒/篇\n")
            f.write(f"平均Token消耗: {avg_tokens:,.0f} tokens/篇\n")
            f.write(f"预计处理速度: {3600 / avg_time:.1f} 篇/小时\n")
            f.write("\n" + "=" * 80 + "\n\n")

        logger.info(f"统计信息已写入: {stats_file}")

    def _load_statistics_from_file(self):
        """从paper_processing_stats.txt加载历史统计数据"""
        stats_file = os.path.join(self.working_dir, "paper_processing_stats.txt")
        
        if not os.path.exists(stats_file):
            logger.info("未找到历史统计文件，从零开始")
            return
        
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析每篇论文的统计信息
            paper_blocks = content.split("论文 #")[1:]  # 跳过表头
            
            for block in paper_blocks:
                lines = block.strip().split('\n')
                
                paper_id = None
                processing_time = None
                tokens_used = None
                
                for line in lines:
                    if line.startswith("论文ID:"):
                        # 提取论文ID
                        paper_id = line.split(":", 1)[1].strip()
                    
                    elif line.startswith("处理时间:"):
                        # 提取 "处理时间: 123.45 秒"
                        time_str = line.split(":")[1].strip().split()[0]
                        try:
                            processing_time = float(time_str)
                        except:
                            pass
                    
                    elif line.startswith("Token消耗:"):
                        # 提取 "Token消耗: 12,345"
                        token_str = line.split(":")[1].strip().replace(',', '')
                        try:
                            tokens_used = int(token_str)
                        except:
                            pass
                
                if paper_id and processing_time is not None and tokens_used is not None:
                    self._stats["total_papers_processed"] += 1
                    self._stats["total_time_seconds"] += processing_time
                    self._stats["total_tokens_used"] += tokens_used
                    self._stats["paper_tokens"].append(tokens_used)
                    self._stats["processed_paper_ids"].add(paper_id)  # 记录已处理的ID
            
            logger.info(f"已加载历史统计: {self._stats['total_papers_processed']} 篇论文, "
                       f"总时间 {self._stats['total_time_seconds']:.2f}秒, "
                       f"总Token {self._stats['total_tokens_used']:,}")
        
        except Exception as e:
            logger.warning(f"加载历史统计文件失败: {str(e)}")


if __name__ == "__main__":
    rag = MetaAnalysisGraphRAG()
    md_dir = "D:/YJS/TMSrag/rTMS-rag/process/dataset/rTMS/markdown/*"
    file_path_list = glob.glob(md_dir + "/*-with-image-refs.md")
    print(f"找到 {len(file_path_list)} 个MD文件")
    
    for file_path in file_path_list:
        print(f"处理文件: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            rag.insert_papers(f.read())
    
    # 打印统计信息
    rag.print_statistics()
    
    stats = rag.get_statistics()
    print(f"\n详细统计: {stats}")



