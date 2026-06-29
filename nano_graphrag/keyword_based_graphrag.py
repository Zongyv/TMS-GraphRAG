import asyncio
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, cast
from collections import defaultdict
import re
import numpy as np
from nano_graphrag.meta_analysis_graphrag import MetaAnalysisGraphRAG
from nano_graphrag._utils import logger, compute_mdhash_id, always_get_an_event_loop
from nano_graphrag.base import BaseKVStorage, BaseGraphStorage, StorageNameSpace
from nano_graphrag._storage import JsonKVStorage, NetworkXStorage
from nano_graphrag.prompt import PROMPTS
from meta_analysis import MetaAnalyzer
from outcome_aliases_config import (
    BRAIN_REGIONS,
    DISEASE_CONDITIONS,
    INTERVENTION_HIERARCHY,
    INTERVENTION_SYNONYMS,
    STUDY_DESIGNS,
    OUTCOME_ALIASES,
    INTERVENTION_QUERY_KEYWORDS
)


@dataclass
class MultiLayerGraphRAG:
    """多层次连接的研究知识图谱系统"""
    
    working_dir: str = "./meta_analysis_cache"
    
    # 继承Meta分析配置
    meta_graphrag: Optional[MetaAnalysisGraphRAG] = field(default=None, init=False)
    
    # 存储组件
    paper_nodes_storage: Optional[BaseKVStorage] = field(default=None, init=False)
    graph_storage: Optional[BaseGraphStorage] = field(default=None, init=False)
    
    # 关键词和语义索引
    keyword_index: Optional[BaseKVStorage] = field(default=None, init=False)
    semantic_index: Optional[BaseKVStorage] = field(default=None, init=False)
    
    # 社区和聚类
    communities_storage: Optional[BaseKVStorage] = field(default=None, init=False)
    
    # 连接权重配置
    keyword_weight: float = 0.3
    structural_weight: float = 0.4
    semantic_weight: float = 0.3
    
    # 相似度阈值
    keyword_similarity_threshold: float = 0.2
    structural_similarity_threshold: float = 0.3
    semantic_similarity_threshold: float = 0.6
    
    def __post_init__(self):
        """初始化多层次图系统"""
        
        # 初始化Meta分析GraphRAG，使用相同的工作目录
        self.meta_graphrag = MetaAnalysisGraphRAG(
            working_dir=self.working_dir
        )
        
        # 从meta_graphrag获取全局配置并扩展
        global_config = {
            "working_dir": self.working_dir,
            "max_graph_cluster_size": getattr(self.meta_graphrag, 'max_graph_cluster_size', 10),
            "graph_cluster_seed": getattr(self.meta_graphrag, 'graph_cluster_seed', 0xDEADBEEF),

            "leiden_gamma": 2.0,  # 增大gamma获得更多社区（默认1.0）
            "leiden_theta": 0.01,  # 细化阈值
            "leiden_tolerance": 0.0001,  # 收敛容差
            "leiden_max_levels": 5,  # 最大层级数（默认从max_graph_cluster_size推导）

            "node2vec_params": getattr(self.meta_graphrag, 'node2vec_params', {
                "dimensions": 128,
                "walk_length": 40,
                "num_walks": 10,
                "workers": 1
            })
        }
        
        # 使用统一的全局配置初始化所有存储组件
        self.paper_nodes_storage = JsonKVStorage(
            namespace="paper_nodes",
            global_config=global_config
        )
        
        self.graph_storage = NetworkXStorage(
            namespace="research_graph",
            global_config=global_config
        )
        
        self.keyword_index = JsonKVStorage(
            namespace="keyword_index",
            global_config=global_config
        )
        
        self.semantic_index = JsonKVStorage(
            namespace="semantic_index",
            global_config=global_config
        )
        
        self.communities_storage = JsonKVStorage(
            namespace="communities",
            global_config=global_config
        )

    async def is_trained(self) -> bool:
        """检查模型是否已经训练完成"""
        try:
            # 检查是否有论文节点
            paper_keys = await self.paper_nodes_storage.all_keys()
            if not paper_keys or len(paper_keys) == 0:
                return False

            # 检查是否有图结构
            if not hasattr(self.graph_storage, '_graph') or self.graph_storage._graph is None:
                return False

            node_count = self.graph_storage._graph.number_of_nodes()
            if node_count == 0:
                return False

            # 检查是否有关键词索引
            keyword_keys = await self.keyword_index.all_keys()
            if not keyword_keys or len(keyword_keys) == 0:
                return False

            logger.info(f"检测到已训练的模型：{len(paper_keys)} 篇论文，{node_count} 个节点")
            return True

        except Exception as e:
            logger.warning(f"检查训练状态失败: {e}")
            return False

    async def get_training_status(self) -> dict:
        """获取训练状态信息"""
        try:
            paper_keys = await self.paper_nodes_storage.all_keys()
            paper_count = len(paper_keys) if paper_keys else 0

            node_count = 0
            edge_count = 0
            if hasattr(self.graph_storage, '_graph') and self.graph_storage._graph is not None:
                node_count = self.graph_storage._graph.number_of_nodes()
                edge_count = self.graph_storage._graph.number_of_edges()

            keyword_keys = await self.keyword_index.all_keys()
            keyword_count = len(keyword_keys) if keyword_keys else 0

            semantic_keys = await self.semantic_index.all_keys()
            semantic_count = len(semantic_keys) if semantic_keys else 0

            return {
                "is_trained": await self.is_trained(),
                "paper_count": paper_count,
                "node_count": node_count,
                "edge_count": edge_count,
                "keyword_count": keyword_count,
                "semantic_count": semantic_count
            }
        except Exception as e:
            logger.error(f"获取训练状态失败: {e}")
            return {"is_trained": False, "error": str(e)}

    async def insert_paper(self, paper_content: str) -> str:
        """插入论文并构建多层次知识图谱"""

        await self._insert_start()
        try:
            logger.info("开始处理论文并构建多层次图...")

            # 1. 基础元数据提取 - 使用异步方法
            await self.meta_graphrag.ainsert_papers(paper_content)
            paper_id = await self._extract_paper_id(paper_content)

            # 2. 构建论文节点
            paper_node = await self._build_paper_node(paper_content, paper_id)

            # 3. 存储论文节点
            await self.paper_nodes_storage.upsert({paper_id: paper_node})
            await self.graph_storage.upsert_node(paper_id, paper_node)

            # 4. 构建多层次连接
            await self._build_multilayer_connections(paper_id, paper_node)

            # 5. 更新索引
            await self._update_indexes(paper_id, paper_node)

            # 6. 增量更新社区
            await self._incremental_community_update(paper_id)

            logger.info(f"论文处理完成: {paper_id}")
            return paper_id
        finally:
            await self._insert_done()

    async def build_graph_from_existing_data(self) -> dict:
        """从已有的evaluated_papers数据构建图，不重新处理论文"""

        logger.info("开始从已有数据构建图...")

        await self._insert_start()
        try:
            # 1. 获取所有已评估的论文
            all_paper_ids = await self.meta_graphrag.evaluated_papers_storage.all_keys()

            if not all_paper_ids:
                logger.warning("未找到已评估的论文数据")
                return {"success": False, "message": "未找到已评估的论文数据"}

            logger.info(f"找到 {len(all_paper_ids)} 篇已评估论文")

            # 2. 逐个构建论文节点和图结构
            processed_count = 0
            failed_papers = []

            for paper_id in all_paper_ids:
                try:
                    # 检查是否已处理
                    existing_node = await self.paper_nodes_storage.get_by_id(paper_id)
                    if existing_node:
                        logger.info(f"论文 {paper_id} 已存在，跳过")
                        continue

                    # 构建论文节点（不需要paper_content）
                    paper_node = await self._build_paper_node_from_metadata(paper_id)

                    # 存储论文节点
                    await self.paper_nodes_storage.upsert({paper_id: paper_node})
                    await self.graph_storage.upsert_node(paper_id, paper_node)

                    # 更新索引
                    await self._update_indexes(paper_id, paper_node)

                    processed_count += 1
                    logger.info(f"处理进度: {processed_count}/{len(all_paper_ids)}")

                except Exception as e:
                    logger.error(f"处理论文 {paper_id} 失败: {e}")
                    failed_papers.append(paper_id)
                    continue

            # 3. 批量构建所有论文之间的连接
            logger.info("开始构建论文间的多层次连接...")
            await self._build_all_multilayer_connections()

            # 4. 执行社区检测
            logger.info("开始社区检测...")
            await self._incremental_community_update(None)

            result = {
                "success": True,
                "total_papers": len(all_paper_ids),
                "processed": processed_count,
                "failed": len(failed_papers),
                "failed_papers": failed_papers
            }

            logger.info(f"图构建完成: {result}")
            return result

        finally:
            await self._insert_done()

    async def _build_paper_node_from_metadata(self, paper_id: str) -> dict:
        """仅从元数据构建论文节点，不需要原始论文内容"""

        # 获取Meta分析的元数据
        meta_data = await self.meta_graphrag.evaluated_papers_storage.get_by_id(paper_id)

        if not meta_data:
            raise ValueError(f"未找到论文 {paper_id} 的元数据")

        if isinstance(meta_data, str):
            logger.error(f"meta_data是字符串而不是字典: {meta_data[:100]}")
            import json
            try:
                meta_data = json.loads(meta_data)
            except:
                raise ValueError(f"无法解析meta_data为JSON")

        if not isinstance(meta_data, dict):
            raise ValueError(f"meta_data类型错误: {type(meta_data)}")

        # 从raw_papers_storage获取摘要和全文
        abstract_content, fulltext_content = await self._separate_abstract_and_fulltext(paper_id)

        # 提取关键词
        keywords = await self._extract_comprehensive_keywords(abstract_content, fulltext_content)

        # 提取结构化特征
        structured_features = await self._extract_structured_features(meta_data)

        # 计算摘要embedding
        abstract_embedding = await self._compute_text_embedding(abstract_content)

        # 构建节点
        paper_node = {
            "paper_id": paper_id,
            "title": meta_data.get("title", "Unknown"),
            "abstract": abstract_content,
            "keywords": keywords,
            "structured_features": structured_features,
            "abstract_embedding": abstract_embedding,
            "quality_score": meta_data.get("quality_assessment", {}).get("overall_score", 0),
            "insert_time": asyncio.get_event_loop().time()
        }

        return paper_node

    async def _build_all_multilayer_connections(self):
        """批量构建所有论文之间的多层次连接"""

        # 获取所有论文节点
        all_paper_ids = await self.paper_nodes_storage.all_keys()

        if not all_paper_ids or len(all_paper_ids) < 2:
            logger.warning("论文数量不足，无法构建连接")
            return

        all_papers = {}
        paper_values = await self.paper_nodes_storage.get_by_ids(all_paper_ids)
        all_papers = {key: value for key, value in zip(all_paper_ids, paper_values) if value is not None}

        logger.info(f"开始为 {len(all_papers)} 篇论文构建连接...")

        # 遍历所有论文对
        total_pairs = len(all_papers) * (len(all_papers) - 1) // 2
        processed_pairs = 0

        paper_ids_list = list(all_papers.keys())
        for i, paper1_id in enumerate(paper_ids_list):
            for paper2_id in paper_ids_list[i + 1:]:
                # 计算相似度
                similarities = await self._calculate_multilayer_similarity(
                    all_papers[paper1_id],
                    all_papers[paper2_id]
                )

                # 计算总权重
                total_weight = (
                        similarities["keyword"] * self.keyword_weight +
                        similarities["structural"] * self.structural_weight +
                        similarities["semantic"] * self.semantic_weight
                )

                # 创建边
                if total_weight > 0.1:  # 最低阈值
                    await self._create_edge(paper1_id, paper2_id, similarities, total_weight)

                processed_pairs += 1
                if processed_pairs % 100 == 0:
                    logger.info(f"连接构建进度: {processed_pairs}/{total_pairs}")

        logger.info(f"完成 {processed_pairs} 对论文的连接构建")

    async def _insert_start(self):
        """开始插入操作的回调"""
        tasks = []
        for storage_inst in [
            self.graph_storage,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        if tasks:
            await asyncio.gather(*tasks)

    async def _insert_done(self):
        """完成插入操作的回调，确保数据持久化"""
        tasks = []
        for storage_inst in [
            self.paper_nodes_storage,
            self.graph_storage,
            self.keyword_index,
            self.semantic_index,
            self.communities_storage,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        if tasks:
            await asyncio.gather(*tasks)

    async def _build_paper_node(self, paper_content: str, paper_id: str) -> dict:
        """构建论文节点数据结构"""

        # 获取Meta分析的元数据
        meta_data = await self.meta_graphrag.evaluated_papers_storage.get_by_id(paper_id)

        # 分离摘要和全文
        abstract_content, fulltext_content = await self._separate_abstract_and_fulltext(paper_id)

        # 提取关键词
        keywords = await self._extract_comprehensive_keywords(abstract_content, fulltext_content)

        # 提取结构化特征
        structured_features = await self._extract_structured_features(meta_data)

        # 计算摘要embedding
        abstract_embedding = await self._compute_text_embedding(abstract_content)

        # 构建节点
        paper_node = {
            "paper_id": paper_id,
            "title": meta_data.get("title", "Unknown") if meta_data else "Unknown",
            "abstract": abstract_content,
            "keywords": keywords,
            "structured_features": structured_features,
            "abstract_embedding": abstract_embedding,
            "quality_score": meta_data.get("quality_assessment", {}).get("overall_score", 0) if meta_data else 0,
            "insert_time": asyncio.get_event_loop().time()
        }

        return paper_node

    async def _extract_comprehensive_keywords(self, abstract: str, fulltext: str) -> dict:
        """提取分层关键词 - 使用LLM直接分类"""

        # 分别从摘要和全文提取分类关键词
        abstract_keywords = await self._extract_categorized_keywords_from_text(abstract, "摘要")
        fulltext_keywords = await self._extract_categorized_keywords_from_text(fulltext, "全文")

        # 合并同类关键词并去重
        merged_keywords = {
            "population": [],      # 人群特征
            "intervention": [],    # 干预方法
            "outcome": [],         # 结局指标
            "design": [],          # 研究设计
            "general": []          # 一般关键词
        }

        # 合并摘要和全文的关键词
        for category in merged_keywords.keys():
            abstract_kw = set(abstract_keywords.get(category, []))
            fulltext_kw = set(fulltext_keywords.get(category, []))
            merged_keywords[category] = list(abstract_kw | fulltext_kw)

        return merged_keywords

    async def _extract_categorized_keywords_from_text(self, text: str, text_type: str) -> dict:
        """从文本中提取已分类的关键词"""

        if not text.strip():
            return self._get_empty_keyword_categories()

        keyword_prompt = PROMPTS["keyword_prompt"].format(text_type=text_type, text=text)

        try:
            response = await self.meta_graphrag.cheap_model_func(keyword_prompt)
            keywords_dict = self._parse_json_response(response)
            return self._clean_keywords(keywords_dict)

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return self._get_empty_keyword_categories()

    def _parse_json_response(self, response: str) -> dict:
        """解析JSON响应"""
        import json

        # 提取JSON部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        raise ValueError("未找到有效JSON")

    def _clean_keywords(self, keywords_dict: dict) -> dict:
        """清理关键词"""
        cleaned = self._get_empty_keyword_categories()

        for category, keywords in keywords_dict.items():
            if category in cleaned and isinstance(keywords, list):
                clean_list = []
                for kw in keywords:
                    if isinstance(kw, str) and 1 < len(kw.strip()) < 50:
                        clean_list.append(kw.strip().lower())
                cleaned[category] = clean_list

        return cleaned

    def _get_empty_keyword_categories(self) -> dict:
        """获取空的关键词分类字典"""
        return {
            "population": [],
            "intervention": [],
            "outcome": [],
            "design": [],
            "general": []
        }

    async def _extract_structured_features(self, meta_data: dict) -> dict:
        """提取结构化特征"""

        if not meta_data:
            return self._get_default_structured_features()

        # 从baseline_characteristics获取基线特征
        baseline = meta_data.get("baseline_characteristics", {})

        # 检查是否为多臂实验
        is_multi_arm = "intervention_groups" in meta_data

        if is_multi_arm:
            # 多臂实验：合并所有分组的干预信息
            return self._extract_multi_arm_structured_features(meta_data, baseline)
        else:
            # 标准双臂实验
            return self._extract_standard_structured_features(meta_data, baseline)

    def _extract_standard_structured_features(self, meta_data: dict, baseline: dict) -> dict:
        """提取标准双臂实验的结构化特征"""
        
        structured_features = {
            "participants": {
                "condition": meta_data.get("population", ""),
                "sample_size": meta_data.get("sample_size", 0),
                "age_range": baseline.get("mean_age", "") if baseline else "",
                "gender_ratio": self._extract_gender_info(baseline) if baseline else ""
            },
            "intervention": {
                "type": meta_data.get("tms_type", ""),
                "frequency": meta_data.get("stimulation_frequency", ""),
                "intensity": meta_data.get("stimulation_intensity", ""),
                "target": meta_data.get("brain_target", ""),
                "sessions": meta_data.get("session_number", ""),
                "duration": meta_data.get("train_duration", ""),
                "hemisphere": meta_data.get("hemisphere", ""),
                "pulses_per_session": meta_data.get("total_number_pulses_per_session", ""),
                "sham_type": meta_data.get("sham_type", "")
            },
            "outcomes": {
                "primary": [o.get("outcome_name", "") for o in meta_data.get("primary_outcomes", [])],
                "secondary": [o.get("outcome_name", "") for o in meta_data.get("secondary_outcomes", [])],
                "measures": self._extract_outcome_measures(meta_data)
            },
            "design": {
                "study_type": meta_data.get("study_type", ""),
                "blinding": self._extract_blinding_info(meta_data),
                "randomization": self._extract_randomization_info(meta_data),
                "dropout_rate": baseline.get("dropout_rate", 0) if baseline else 0
            }
        }
        
        return structured_features

    def _extract_multi_arm_structured_features(self, meta_data: dict, baseline: dict) -> dict:
        """提取多臂实验的结构化特征（合并所有分组信息）"""

        intervention_groups = meta_data.get("intervention_groups", [])
        control_group = meta_data.get("control_group", {})

        # 过滤出有TMS干预的组
        tms_groups = [g for g in intervention_groups if g.get("tms_type")]

        if not tms_groups:
            logger.warning("多臂实验中没有找到TMS干预组，使用默认值")
            return self._get_default_structured_features()

        # 合并所有TMS干预组的参数
        all_types = []
        all_frequencies = []
        all_intensities = []
        all_targets = []
        all_sessions = []
        all_durations = []
        all_hemispheres = []
        all_pulses = []

        for group in tms_groups:
            if group.get("tms_type") and group["tms_type"] not in all_types:
                all_types.append(group["tms_type"])

            if group.get("stimulation_frequency"):
                freq_str = str(group["stimulation_frequency"])
                if freq_str not in all_frequencies:
                    all_frequencies.append(freq_str)

            if group.get("stimulation_intensity"):
                intensity_str = str(group["stimulation_intensity"])
                if intensity_str not in all_intensities:
                    all_intensities.append(intensity_str)

            if group.get("brain_target") and group["brain_target"] not in all_targets:
                all_targets.append(group["brain_target"])

            if group.get("session_number"):
                session_str = str(group["session_number"])
                if session_str not in all_sessions:
                    all_sessions.append(session_str)

            if group.get("train_duration"):
                duration_str = str(group["train_duration"])
                if duration_str not in all_durations:
                    all_durations.append(duration_str)

            if group.get("hemisphere") and group["hemisphere"] not in all_hemispheres:
                all_hemispheres.append(group["hemisphere"])

            if group.get("total_number_pulses_per_session"):
                pulses_str = str(group["total_number_pulses_per_session"])
                if pulses_str not in all_pulses:
                    all_pulses.append(pulses_str)

        structured_features = {
            "participants": {
                "condition": meta_data.get("population", ""),
                "sample_size": meta_data.get("sample_size", 0),
                "age_range": baseline.get("mean_age", "") if baseline else "",
                "gender_ratio": self._extract_gender_info(baseline) if baseline else ""
            },
            "intervention": {
                "type": ", ".join(all_types) if all_types else "",
                "frequency": ", ".join(all_frequencies) if all_frequencies else "",
                "intensity": ", ".join(all_intensities) if all_intensities else "",
                "target": ", ".join(all_targets) if all_targets else "",
                "sessions": ", ".join(all_sessions) if all_sessions else "",
                "duration": ", ".join(all_durations) if all_durations else "",
                "hemisphere": ", ".join(all_hemispheres) if all_hemispheres else "",
                "pulses_per_session": ", ".join(all_pulses) if all_pulses else "",
                "sham_type": control_group.get("sham_type", "") if control_group else ""
            },
            "outcomes": {
                "primary": [o.get("outcome_name", "") for o in meta_data.get("primary_outcomes", [])],
                "secondary": [o.get("outcome_name", "") for o in meta_data.get("secondary_outcomes", [])],
                "measures": self._extract_outcome_measures(meta_data)
            },
            "design": {
                "study_type": meta_data.get("study_type", ""),
                "blinding": self._extract_blinding_info(meta_data),
                "randomization": self._extract_randomization_info(meta_data),
                "dropout_rate": baseline.get("dropout_rate", 0) if baseline else 0
            }
        }

        logger.info(f"多臂实验：总组数={len(intervention_groups)}, TMS组数={len(tms_groups)}")
        return structured_features

    def _extract_gender_info(self, baseline: dict) -> str:
        """从基线特征中提取性别信息"""

        gender_dist = baseline.get("gender_distribution", {})
        male_percent = gender_dist.get("male_percent")
        female_percent = gender_dist.get("female_percent")

        if male_percent is not None and female_percent is not None:
            return f"Male: {male_percent}%, Female: {female_percent}%"
        elif male_percent is not None:
            return f"Male: {male_percent}%"
        elif female_percent is not None:
            return f"Female: {female_percent}%"
        else:
            return ""

    def _extract_outcome_measures(self, meta_data: dict) -> list:
        """提取所有结局测量工具"""

        measures = []

        # 从outcome_definitions中提取
        outcome_defs = meta_data.get("outcome_definitions", {})

        # 主要结局指标的测量工具
        for outcome in outcome_defs.get("primary_outcomes", []):
            scale = outcome.get("scale", "")
            if scale and scale not in measures:
                measures.append(scale)

        # 次要结局指标的测量工具
        for outcome in outcome_defs.get("secondary_outcomes", []):
            scale = outcome.get("scale", "")
            if scale and scale not in measures:
                measures.append(scale)

        return measures

    def _extract_blinding_info(self, meta_data: dict) -> str:
        """从RoB2评估中提取盲法信息"""

        rob2 = meta_data.get("rob2_assessment", {})
        domain2 = rob2.get("domain2_deviations", {})

        # 从rationale中推断盲法类型
        rationale = domain2.get("rationale", "").lower()
        evidence = domain2.get("supporting_evidence", "").lower()

        if "双盲" in rationale or "double-blind" in evidence:
            return "double-blind"
        elif "单盲" in rationale or "single-blind" in evidence:
            return "single-blind"
        elif "盲" in rationale or "blind" in evidence:
            return "blinded"
        else:
            return ""

    def _extract_randomization_info(self, meta_data: dict) -> str:
        """从RoB2评估中提取随机化信息"""

        rob2 = meta_data.get("rob2_assessment", {})
        domain1 = rob2.get("domain1_randomization", {})

        # 从rationale和evidence中提取随机化方法
        rationale = domain1.get("rationale", "").lower()
        evidence = domain1.get("supporting_evidence", "").lower()

        if "随机数字生成器" in rationale or "random number generator" in evidence:
            return "random number generator"
        elif "随机" in rationale or "random" in evidence:
            return "randomized"
        else:
            return ""

    def _get_default_structured_features(self) -> dict:
        """获取默认结构化特征"""

        return {
            "participants": {
                "condition": "",
                "sample_size": 0,
                "age_range": "",
                "gender_ratio": ""
            },
            "intervention": {
                "type": "",
                "frequency": "",
                "intensity": "",
                "target": "",
                "sessions": "",
                "duration": "",
                "hemisphere": "",
                "pulses_per_session": ""
            },
            "outcomes": {
                "primary": [],
                "secondary": [],
                "measures": []
            },
            "design": {
                "study_type": "",
                "blinding": "",
                "randomization": "",
                "dropout_rate": 0
            }
        }

    async def _compute_text_embedding(self, text: str) -> List[float]:
        """计算文本embedding"""

        try:
            # 使用meta_graphrag的embedding函数
            embedding = await self.meta_graphrag.embedding_func([text])

            # 检查embedding是否为None
            if embedding is None:
                return []

            # 处理numpy数组格式
            if isinstance(embedding, np.ndarray):
                if embedding.size == 0:
                    return []
                # 取第一行并转换为列表
                return embedding[0].tolist()
            else:
                logger.warning(f"Embedding函数返回未知类型: {type(embedding)}")
                return []

        except Exception as e:
            logger.error(f"Embedding计算失败: {e}")
            return []

    async def _build_multilayer_connections(self, paper_id: str, paper_node: dict):
        """构建多层次连接"""

        # 获取所有现有论文节点
        all_paper_ids = await self.paper_nodes_storage.all_keys()
        all_papers = {}

        if all_paper_ids:
            paper_nodes = await self.paper_nodes_storage.get_by_ids(all_paper_ids)
            all_papers = {paper_id: node for paper_id, node in zip(all_paper_ids, paper_nodes) if node is not None}

        for existing_id, existing_node in all_papers.items():
            if existing_id == paper_id:
                continue

            # 计算多层次相似度
            similarities = await self._calculate_multilayer_similarity(paper_node, existing_node)

            # 计算综合权重
            total_weight = (
                similarities["keyword"] * self.keyword_weight +
                similarities["structural"] * self.structural_weight +
                similarities["semantic"] * self.semantic_weight
            )

            # 如果相似度超过阈值，创建连接
            if total_weight > 0.3:  # 总体阈值
                await self._create_edge(paper_id, existing_id, similarities, total_weight)

    async def _calculate_multilayer_similarity(self, node1: dict, node2: dict) -> dict:
        """计算多层次相似度"""

        similarities = {
            "keyword": 0.0,
            "structural": 0.0,
            "semantic": 0.0
        }

        # 1. 关键词相似度
        similarities["keyword"] = self._calculate_keyword_similarity(
            node1["keywords"], node2["keywords"]
        )

        # 2. 结构化特征相似度
        similarities["structural"] = self._calculate_structural_similarity(
            node1["structured_features"], node2["structured_features"]
        )

        # 3. 语义相似度
        if node1["abstract_embedding"] and node2["abstract_embedding"]:
            similarities["semantic"] = self._calculate_semantic_similarity(
                node1["abstract_embedding"], node2["abstract_embedding"]
            )

        return similarities

    def _calculate_keyword_similarity(self, keywords1: dict, keywords2: dict) -> float:
        """计算关键词相似度"""

        total_similarity = 0.0
        category_weights = {
            "population": 0.3,
            "intervention": 0.3,
            "outcome": 0.25,
            "design": 0.1,
            "general": 0.05
        }

        for category, weight in category_weights.items():
            kw1 = set(keywords1.get(category, []))
            kw2 = set(keywords2.get(category, []))

            if not kw1 and not kw2:
                continue

            # Jaccard相似度
            intersection = len(kw1 & kw2)
            union = len(kw1 | kw2)

            if union > 0:
                jaccard = intersection / union
                total_similarity += jaccard * weight

        return total_similarity

    def _calculate_structural_similarity(self, features1: dict, features2: dict) -> float:
        """计算结构化特征相似度"""

        total_similarity = 0.0
        feature_weights = {
            "participants": 0.3,
            "intervention": 0.4,
            "outcomes": 0.2,
            "design": 0.1
        }

        for feature_type, weight in feature_weights.items():
            f1 = features1.get(feature_type, {})
            f2 = features2.get(feature_type, {})

            if feature_type == "participants":
                sim = self._compare_participants(f1, f2)
            elif feature_type == "intervention":
                sim = self._compare_interventions(f1, f2)
            elif feature_type == "outcomes":
                sim = self._compare_outcomes(f1, f2)
            elif feature_type == "design":
                sim = self._compare_designs(f1, f2)
            else:
                sim = 0.0

            total_similarity += sim * weight

        return total_similarity

    def _compare_participants(self, p1: dict, p2: dict) -> float:
        """比较被试特征"""

        similarity = 0.0
        total_weight = 0.0  #跟踪实际比较的维度

        # 疾病条件匹配 (权重: 0.5)
        if p1.get("condition") and p2.get("condition"):
            total_weight += 0.5
            if p1["condition"].lower() == p2["condition"].lower():
                similarity += 0.5
            elif any(term in p2["condition"].lower() for term in p1["condition"].lower().split()):
                similarity += 0.3

        # 样本量相似性 (权重: 0.3) - 只有两者都有值时才比较
        size1 = p1.get("sample_size")
        size2 = p2.get("sample_size")

        # 转换为整数，但保留None
        if size1 is not None:
            try:
                size1 = int(size1)
            except (ValueError, TypeError):
                size1 = None

        if size2 is not None:
            try:
                size2 = int(size2)
            except (ValueError, TypeError):
                size2 = None

        # 只有两者都有有效值时才比较
        if size1 is not None and size2 is not None and size1 > 0 and size2 > 0:
            total_weight += 0.3
            size_ratio = min(size1, size2) / max(size1, size2)
            similarity += 0.3 * size_ratio

        # 年龄范围相似性 (权重: 0.2)
        if p1.get("age_range") and p2.get("age_range"):
            total_weight += 0.2
            if p1["age_range"] == p2["age_range"]:
                similarity += 0.2

        # 归一化：根据实际比较的维度计算相似度
        if total_weight > 0:
            return min(similarity / total_weight, 1.0)
        else:
            return 0.0  # 没有可比较的维度

    def _compare_interventions(self, i1: dict, i2: dict) -> float:
        """比较干预方法"""

        similarity = 0.0
        total_weight = 0.0  #跟踪实际比较的维度

        # 干预类型 (权重: 0.4)
        if i1.get("type") and i2.get("type"):
            total_weight += 0.4
            if i1["type"].lower() == i2["type"].lower():
                similarity += 0.4
            elif any(term in i2["type"].lower() for term in i1["type"].lower().split()):
                similarity += 0.2

        # 刺激频率 (权重: 0.2)
        if i1.get("frequency") and i2.get("frequency"):
            total_weight += 0.2
            if i1["frequency"] == i2["frequency"]:
                similarity += 0.2

        # 刺激靶点 (权重: 0.2)
        if i1.get("target") and i2.get("target"):
            total_weight += 0.2
            if i1["target"].lower() == i2["target"].lower():
                similarity += 0.2
            elif any(term in i2["target"].lower() for term in i1["target"].lower().split()):
                similarity += 0.1

        # 治疗次数相似性 (权重: 0.2) - 只有两者都有值时才比较
        sessions1 = self._safe_get_int(i1, "sessions")
        sessions2 = self._safe_get_int(i2, "sessions")

        if sessions1 is not None and sessions2 is not None and sessions1 > 0 and sessions2 > 0:
            total_weight += 0.2
            sessions_ratio = min(sessions1, sessions2) / max(sessions1, sessions2)
            similarity += 0.2 * sessions_ratio

        # 归一化
        if total_weight > 0:
            return min(similarity / total_weight, 1.0)
        else:
            return 0.0

    def _safe_get_int(self, data: dict, key: str) -> int:
        """安全地从字典中获取整数值，如果不存在或无效则返回None"""
        value = data.get(key)

        if value is None or value == "":
            return None

        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except (ValueError, TypeError):
                return None
        elif isinstance(value, float):
            return int(value)
        else:
            return None

    def _compare_outcomes(self, o1: dict, o2: dict) -> float:
        """比较结局指标"""

        primary1 = set(o1.get("primary", []))
        primary2 = set(o2.get("primary", []))

        if not primary1 and not primary2:
            return 0.0

        # 主要结局指标的Jaccard相似度
        intersection = len(primary1 & primary2)
        union = len(primary1 | primary2)

        return intersection / union if union > 0 else 0.0

    def _compare_designs(self, d1: dict, d2: dict) -> float:
        """比较研究设计"""

        similarity = 0.0

        # 研究类型
        if d1.get("study_type") and d2.get("study_type"):
            if d1["study_type"].lower() == d2["study_type"].lower():
                similarity += 0.5

        # 盲法
        if d1.get("blinding") and d2.get("blinding"):
            if d1["blinding"].lower() == d2["blinding"].lower():
                similarity += 0.3

        # 随机化方法
        if d1.get("randomization") and d2.get("randomization"):
            if d1["randomization"].lower() == d2["randomization"].lower():
                similarity += 0.2

        return min(similarity, 1.0)

    def _calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算语义相似度（余弦相似度）"""

        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # 余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            cosine_sim = dot_product / (norm1 * norm2)
            return max(0.0, cosine_sim)  # 确保非负

        except Exception as e:
            logger.error(f"语义相似度计算失败: {e}")
            return 0.0

    async def _create_edge(self, paper1_id: str, paper2_id: str, similarities: dict, total_weight: float):
        """创建图边"""

        # 确保权重是有效的数值
        if total_weight <= 0 or not isinstance(total_weight, (int, float)):
            logger.warning(f"无效的边权重: {total_weight}")
            return

        edge_data = {
            "weight": float(total_weight),  # 确保是float类型
            "keyword_similarity": float(similarities["keyword"]),
            "structural_similarity": float(similarities["structural"]),
            "semantic_similarity": float(similarities["semantic"]),
            "edge_type": self._determine_edge_type(similarities),
            "created_time": asyncio.get_event_loop().time()
        }

        try:
            await self.graph_storage.upsert_edge(paper1_id, paper2_id, edge_data)
            logger.debug(f"创建边: {paper1_id} -> {paper2_id}, 权重: {total_weight}")
        except Exception as e:
            logger.error(f"创建边失败: {e}")

    def _determine_edge_type(self, similarities: dict) -> str:
        """确定边的类型"""

        max_sim_type = max(similarities.items(), key=lambda x: x[1])

        if max_sim_type[1] > 0.5:
            return f"strong_{max_sim_type[0]}"
        elif max_sim_type[1] > 0.3:
            return f"moderate_{max_sim_type[0]}"
        else:
            return f"weak_{max_sim_type[0]}"

    async def _update_indexes(self, paper_id: str, paper_node: dict):
        """更新各种索引"""

        # 更新关键词索引
        await self._update_keyword_index(paper_id, paper_node["keywords"])

        # 更新语义索引
        if paper_node["abstract_embedding"]:
            await self._update_semantic_index(paper_id, paper_node["abstract_embedding"])

    async def _update_keyword_index(self, paper_id: str, keywords: dict):
        """更新关键词索引"""

        # 获取当前索引的所有键
        all_keys = await self.keyword_index.all_keys()
        current_index = {}

        if all_keys:
            index_values = await self.keyword_index.get_by_ids(all_keys)
            current_index = {key: value for key, value in zip(all_keys, index_values) if value is not None}

        for category, kw_list in keywords.items():
            for keyword in kw_list:
                if keyword not in current_index:
                    current_index[keyword] = {
                        "papers": [],
                        "category": category,
                        "frequency": 0
                    }

                if paper_id not in current_index[keyword]["papers"]:
                    current_index[keyword]["papers"].append(paper_id)
                    current_index[keyword]["frequency"] += 1

        await self.keyword_index.upsert(current_index)

    async def _update_semantic_index(self, paper_id: str, embedding: List[float]):
        """更新语义索引"""

        # 确保embedding是列表格式
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        semantic_data = {
            paper_id: {
                "embedding": embedding,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        await self.semantic_index.upsert(semantic_data)

    async def _incremental_community_update(self, paper_id: str):
        """增量更新社区结构"""

        try:
            # 检查图是否有足够的节点进行聚类
            if not hasattr(self.graph_storage, '_graph') or self.graph_storage._graph is None:
                logger.warning("图对象不存在，跳过社区检测")
                return

            graph = self.graph_storage._graph
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()

            if node_count < 2:
                logger.info(f"图中只有 {node_count} 个节点，跳过社区检测")
                return

            if edge_count == 0:
                logger.info("图中没有边，跳过社区检测")
                return

            logger.info(f"开始社区检测：{node_count} 个节点，{edge_count} 条边")

            # 执行聚类
            await self.graph_storage.clustering("leiden")

            await self.graph_storage.index_done_callback()
            logger.info("已保存包含clusters的图数据到GraphML")

            # 获取社区信息
            try:
                community_schema = await self.graph_storage.community_schema()
                logger.info(f"成功获取社区信息，发现 {len(community_schema)} 个社区")
            except Exception as community_error:
                logger.error(f"获取社区信息失败: {str(community_error)}")
                import traceback
                logger.error(f"社区信息获取详细错误: {traceback.format_exc()}")
                return

            # 存储社区信息
            try:
                await self.communities_storage.upsert({"communities": community_schema})
                logger.info(f"社区检测完成，发现 {len(community_schema)} 个社区")
                await self.communities_storage.index_done_callback()
                logger.info("已保存社区信息到JSON文件")
            except Exception as storage_error:
                logger.error(f"存储社区信息失败: {str(storage_error)}")

        except Exception as e:
            logger.warning(f"社区检测失败: {str(e)}")
            # 打印更详细的错误信息用于调试
            import traceback
            logger.debug(f"社区检测详细错误: {traceback.format_exc()}")


    """
    前面为知识图谱的构建过程，在此之后为用户查询方法
    """

    async def query_papers(self, query: str, mode: str = "comprehensive") -> dict:
        """多模式查询论文"""

        logger.info(f"处理查询: {query}, 模式: {mode}")

        if mode == "keyword":
            return await self._keyword_query(query)
        elif mode == "semantic":
            return await self._semantic_query(query)
        elif mode == "structural":
            return await self._structural_query(query)
        elif mode == "community_based":
            return await self._community_based_query(query)
        else:  # comprehensive
            return await self._comprehensive_query(query)

    async def _comprehensive_query(self, query: str) -> dict:
        """
        综合查询 - 整合关键词、语义、结构化三种查询方式
        支持必须匹配关键词、排除关键词和年份范围功能
        """

        # 0. 预先验证查询相关性
        relevance_check = await self._check_query_relevance(query)
        if not relevance_check["is_relevant"]:
            return {
                "query": query,
                "mode": "comprehensive",
                "total_papers": 0,
                "papers": [],
                "relevance_warning": relevance_check["reason"],
                "suggested_topics": relevance_check.get("available_topics", [])
            }

        # 1. 关键词查询（基础匹配，包含必须匹配、排除逻辑和年份过滤）
        keyword_results = await self._keyword_query(query)

        # 2. 语义查询（语义相似度）
        semantic_results = await self._semantic_query(query)

        # 3. 结构化查询（细粒度特征匹配）
        structural_results = await self._structural_query(query)

        # 获取通过关键词筛选的论文ID集合
        keyword_qualified_papers = set(p["paper_id"] for p in keyword_results["papers"])

        # 获取必须匹配关键词、排除关键词和年份范围
        must_have_keywords = keyword_results.get("must_have_keywords", [])
        exclude_keywords = keyword_results.get("exclude_keywords", [])
        year_range = keyword_results.get("year_range", {"start": None, "end": None})

        # 4. 结果融合 - 三维度整合
        all_papers = {}

        # 4.1 处理关键词结果
        for paper in keyword_results["papers"]:
            paper_id = paper["paper_id"]
            if await self._verify_paper_relevance(paper_id, query):
                all_papers[paper_id] = {
                    **paper,
                    "keyword_score": paper["keyword_score"],
                    "semantic_score": 0.0,
                    "structural_score": 0.0,
                    "has_keyword_match": True
                }

        # 4.2 处理语义结果
        for paper in semantic_results["papers"]:
            paper_id = paper["paper_id"]
            if await self._verify_paper_relevance(paper_id, query):
                if paper_id in all_papers:
                    all_papers[paper_id]["semantic_score"] = paper["similarity_score"]
                else:
                    all_papers[paper_id] = {
                        **paper,
                        "keyword_score": 0.0,
                        "semantic_score": paper["similarity_score"],
                        "structural_score": 0.0,
                        "has_keyword_match": False
                    }

        # 4.3 处理结构化结果
        for paper in structural_results.get("papers", []):
            paper_id = paper["paper_id"]
            if await self._verify_paper_relevance(paper_id, query):
                if paper_id in all_papers:
                    all_papers[paper_id]["structural_score"] = paper["structural_score"]
                else:
                    all_papers[paper_id] = {
                        **paper,
                        "keyword_score": 0.0,
                        "semantic_score": 0.0,
                        "structural_score": paper["structural_score"],
                        "has_keyword_match": False
                    }

        # 4.4 应用必须匹配关键词过滤（最严格，优先执行）
        if must_have_keywords:
            all_paper_ids = list(all_papers.keys())
            filtered_paper_ids = await self._filter_must_have_keywords(all_paper_ids, must_have_keywords)
            all_papers = {pid: all_papers[pid] for pid in filtered_paper_ids}
            logger.info(f"综合查询必须关键词过滤: {len(all_paper_ids)} -> {len(all_papers)} 篇论文")

        # 4.5 应用年份范围过滤
        if year_range.get("start") or year_range.get("end"):
            all_paper_ids = list(all_papers.keys())
            filtered_paper_ids = await self._filter_by_year_range(all_paper_ids, year_range)
            all_papers = {pid: all_papers[pid] for pid in filtered_paper_ids}
            logger.info(f"综合查询年份范围过滤: {len(all_paper_ids)} -> {len(all_papers)} 篇论文")

        # 4.6 应用排除关键词过滤
        if exclude_keywords:
            all_paper_ids = list(all_papers.keys())
            filtered_paper_ids = await self._filter_out_excluded_papers(all_paper_ids, exclude_keywords)
            all_papers = {pid: all_papers[pid] for pid in filtered_paper_ids}
            logger.info(f"综合查询排除关键词过滤: {len(all_paper_ids)} -> {len(all_papers)} 篇论文")

        # 5. 计算综合得分
        for paper_id, paper in all_papers.items():
            # 基础得分
            base_score = (
                    paper["keyword_score"] * 0.25 +
                    paper["semantic_score"] * 0.4 +
                    paper["structural_score"] * 0.35
            )

            # 如果有关键词匹配，额外加分
            keyword_bonus = 0.1 if paper.get("has_keyword_match", False) else 0.0

            paper["comprehensive_score"] = min(base_score + keyword_bonus, 1.0)

        # 6. 最终过滤
        filtered_papers = {
            pid: paper for pid, paper in all_papers.items()
            if paper["comprehensive_score"] > 0.5 and paper["keyword_score"] > 0
        }

        # 7. 排序
        sorted_papers = sorted(
            filtered_papers.values(),
            key=lambda x: x["comprehensive_score"],
            reverse=True
        )

        return {
            "query": query,
            "mode": "comprehensive",
            "query_keywords": keyword_results.get("query_keywords"),
            "must_have_keywords": must_have_keywords,
            "exclude_keywords": exclude_keywords,
            "year_range": year_range,
            "total_papers": len(sorted_papers),
            "papers": sorted_papers,
            "keyword_matches": keyword_results.get("keyword_matches", {}),
            "semantic_coverage": len(semantic_results["papers"]),
            "structural_coverage": len(structural_results.get("papers", [])),
            "filters_applied": {
                "must_have_keywords": len(must_have_keywords) > 0,
                "year_range": year_range.get("start") or year_range.get("end"),
                "exclude_keywords": len(exclude_keywords) > 0
            }
        }


    async def _check_query_relevance(self, query: str) -> dict:
        """检查查询与知识图谱的相关性"""

        # 1. 提取查询中的核心概念
        query_concepts = await self._extract_core_concepts(query)

        # 2. 获取知识图谱中的主要主题
        graph_topics = await self._get_graph_main_topics()

        # 3. 计算概念重叠度
        concept_overlap = self._calculate_concept_overlap(query_concepts, graph_topics)

        # 4. 判断相关性
        if concept_overlap < 0.1:  # 重叠度过低
            return {
                "is_relevant": False,
                "reason": f"查询主题与知识图谱内容相关性过低 (重叠度: {concept_overlap:.2f})",
                "available_topics": list(graph_topics.keys())[:10],
                "query_concepts": query_concepts
            }

        return {
            "is_relevant": True,
            "overlap_score": concept_overlap,
            "matched_concepts": self._get_matched_concepts(query_concepts, graph_topics)
        }

    async def _extract_core_concepts(self, query: str) -> dict:
        """提取查询中的核心概念 - 智能匹配"""

        concepts = {
            "population": [],
            "intervention": [],
            "outcome": [],
            "general": []
        }

        query_lower = query.lower()

        # 疾病/人群概念
        for disease, keywords in DISEASE_CONDITIONS.items():
            if any(self._match_keyword(kw, query_lower) for kw in keywords):
                concepts["population"].append(disease)

        # 干预概念
        for intervention, keywords in INTERVENTION_QUERY_KEYWORDS.items():
            if any(self._match_keyword(kw, query_lower) for kw in keywords):
                concepts["intervention"].append(intervention)

        # 脑区概念
        for region, keywords in BRAIN_REGIONS.items():
            if any(self._match_keyword(kw, query_lower) for kw in keywords):
                concepts["general"].append(region)

        # 结局指标概念
        matched_outcomes = []
        for outcome, keywords in OUTCOME_ALIASES.items():
            for kw in keywords:
                if self._match_keyword(kw, query_lower):
                    matched_outcomes.append((outcome, len(kw)))
                    break

        # 去重逻辑保持不变
        matched_outcomes.sort(key=lambda x: x[1], reverse=True)
        seen_bases = set()
        for outcome, kw_len in matched_outcomes:
            base_name = outcome.split('-')[0]
            if base_name not in seen_bases:
                concepts["outcome"].append(outcome)
                seen_bases.add(base_name)
            elif outcome != base_name:
                if base_name in concepts["outcome"]:
                    concepts["outcome"].remove(base_name)
                concepts["outcome"].append(outcome)

        return concepts

    def _match_keyword(self, keyword: str, text: str) -> bool:
        """智能关键词匹配 - 短词用边界，长词用包含"""
        if len(keyword) <= 2:
            # 短关键词（如 "ad", "pd"）必须是独立单词
            pattern = r'\b' + re.escape(keyword) + r'\b'
            return bool(re.search(pattern, text))
        else:
            # 长关键词可以部分匹配
            return keyword in text

    async def _get_graph_main_topics(self) -> dict:
        """获取知识图谱中的主要主题"""

        # 获取所有论文的关键词
        paper_keys = await self.paper_nodes_storage.all_keys()
        all_keywords = {"population": set(), "intervention": set(), "outcome": set(), "design": set(), "general": set()}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

            for paper in papers.values():
                keywords = paper.get("keywords", {})

                # 收集各类关键词 - 使用实际的分类
                for category in ["population", "intervention", "outcome", "design", "general"]:
                    if category in keywords and keywords[category]:
                        all_keywords[category].update(keywords[category])

        # 转换为计数字典
        topic_counts = {}
        for category, keyword_set in all_keywords.items():
            for keyword in keyword_set:
                topic_counts[keyword.lower()] = topic_counts.get(keyword.lower(), 0) + 1

        return topic_counts

    def _calculate_concept_overlap(self, query_concepts: dict, graph_topics: dict) -> float:
        """计算概念重叠度 - 通用疾病概念验证"""

        total_query_concepts = sum(len(concepts) for concepts in query_concepts.values())
        if total_query_concepts == 0:
            return 0.0

        matched_concepts = 0

        # 疾病/人群概念必须严格匹配
        population_concepts = query_concepts.get("population", [])
        if population_concepts:
            population_matched = False
            for concept in population_concepts:
                concept_lower = concept.lower()

                # 检查是否有匹配的疾病概念
                for topic in graph_topics.keys():
                    if self._is_disease_match(concept_lower, topic):
                        population_matched = True
                        matched_concepts += 1
                        break

                if population_matched:
                    break

            # 如果查询中有疾病概念但完全不匹配，直接返回低分
            if not population_matched:
                return 0.0  # 疾病不匹配直接判定为不相关

        # 其他概念的匹配
        for category, concepts in query_concepts.items():
            if category == "population":  # 已经处理过
                continue

            for concept in concepts:
                if concept.lower() in graph_topics:
                    matched_concepts += 1
                else:
                    # 其他概念允许部分匹配
                    for topic in graph_topics.keys():
                        if concept.lower() in topic or topic in concept.lower():
                            matched_concepts += 0.3  # 降低部分匹配的权重
                            break

        return matched_concepts / total_query_concepts

    def _is_disease_match(self, query_disease: str, graph_topic: str) -> bool:
        """判断查询疾病与图谱主题是否匹配 - 使用配置文件"""
        return self._are_disease_synonyms(query_disease, graph_topic, DISEASE_CONDITIONS) or \
            self._is_related_disease_term(query_disease, graph_topic)

    async def _verify_paper_relevance(self, paper_id: str, query: str) -> bool:
        """验证单篇论文与查询的相关性"""

        # 获取论文信息
        paper_node = await self.paper_nodes_storage.get_by_id(paper_id)
        if not paper_node:
            return False

        # 提取查询和论文的核心概念
        query_concepts = await self._extract_core_concepts(query)
        paper_keywords = paper_node.get("keywords", {})

        # 检查核心概念匹配
        for category in ["population", "intervention"]:  # 人群和干预是核心概念
            query_items = query_concepts.get(category, [])
            paper_items = paper_keywords.get(category, [])

            if query_items:  # 如果查询中有这类概念
                # 检查是否有匹配
                query_items_lower = [item.lower() for item in query_items]
                paper_items_lower = [item.lower() for item in paper_items]

                # 对于疾病概念，需要更严格的匹配
                if category == "population":
                    has_match = self._check_population_match(query_items_lower, paper_items_lower)
                else:
                    # 干预概念使用层次化匹配
                    has_match = self._check_intervention_match(query_items, paper_items)

                if not has_match:
                    logger.info(f"论文 {paper_id[:30]}... 被拒绝: {category} 不匹配")
                    return False  # 核心概念不匹配，直接排除

        return True

    def _check_population_match(self, query_items: List[str], paper_items: List[str]) -> bool:
        """检查人群/疾病概念匹配 - 通用方法"""

        logger.info(f"    Population匹配检查:")
        logger.info(f"      查询items: {query_items}")
        logger.info(f"      论文items: {paper_items}")

        for q_item in query_items:
            for p_item in paper_items:
                # 1. 完全匹配
                if q_item == p_item:
                    logger.info(f"      完全匹配: '{q_item}' == '{p_item}'")
                    return True

                # 2. 同义词匹配
                if self._are_disease_synonyms(q_item, p_item, DISEASE_CONDITIONS):
                    logger.info(f"      同义词匹配: '{q_item}' ~ '{p_item}'")
                    return True

                # 3. 包含匹配
                if self._is_related_disease_term(q_item, p_item):
                    logger.info(f"      相关词匹配: '{q_item}' <-> '{p_item}'")
                    return True

        return False

    def _check_intervention_match(self, query_items: List[str], paper_items: List[str]) -> bool:
        """检查干预方法匹配 - 支持层次化匹配和同义词匹配"""

        logger.info(f"    Intervention匹配检查:")
        logger.info(f"      查询items: {query_items}")
        logger.info(f"      论文items: {paper_items}")

        for q_item in query_items:
            q_item_lower = q_item.lower()

            for p_item in paper_items:
                p_item_lower = p_item.lower()

                # 1. 直接匹配
                if q_item_lower == p_item_lower:
                    return True

                # 2. 包含匹配
                if q_item_lower in p_item_lower or p_item_lower in q_item_lower:
                    return True

                # 3. 同义词匹配（同级别）
                if q_item_lower in INTERVENTION_SYNONYMS:
                    synonyms = INTERVENTION_SYNONYMS[q_item_lower]
                    if any(syn in p_item_lower for syn in synonyms):
                        return True

                if p_item_lower in INTERVENTION_SYNONYMS:
                    synonyms = INTERVENTION_SYNONYMS[p_item_lower]
                    if any(syn in q_item_lower for syn in synonyms):
                        return True

                # 4. 层次化匹配：如果查询是父类，论文是子类，则匹配
                if q_item_lower in INTERVENTION_HIERARCHY:
                    parent_methods = INTERVENTION_HIERARCHY[q_item_lower]
                    if any(method in p_item_lower for method in parent_methods):
                        return True

                # 5. 反向检查：如果论文是父类，查询是子类，也匹配
                if p_item_lower in INTERVENTION_HIERARCHY:
                    parent_methods = INTERVENTION_HIERARCHY[p_item_lower]
                    if any(method in q_item_lower for method in parent_methods):
                        return True

        return False

    def _are_disease_synonyms(self, term1: str, term2: str, synonyms_dict: dict) -> bool:
        """检查两个术语是否为疾病同义词"""

        term1_lower = term1.lower().strip()
        term2_lower = term2.lower().strip()

        for disease, synonyms in synonyms_dict.items():
            # 将同义词列表转为小写
            synonyms_lower = [s.lower().strip() for s in synonyms]

            # 检查两个术语是否都在同一个同义词组中
            if term1_lower in synonyms_lower and term2_lower in synonyms_lower:
                logger.debug(f"      同义词匹配成功: '{term1}' 和 '{term2}' 都属于 '{disease}'")
                return True

            # 检查疾病名称本身
            if (term1_lower == disease or term1_lower in synonyms_lower) and \
                    (term2_lower == disease or term2_lower in synonyms_lower):
                logger.debug(f"      同义词匹配成功: '{term1}' 和 '{term2}' 都属于 '{disease}'")
                return True

        return False

    def _is_related_disease_term(self, query_term: str, paper_term: str) -> bool:
        """检查是否为相关的疾病术语（更宽松的匹配）"""

        # 检查词根匹配
        if len(query_term) >= 4 and len(paper_term) >= 4:
            # 如果一个词包含另一个词的主要部分
            if query_term in paper_term or paper_term in query_term:
                return True

            # 检查常见的词根
            common_roots = [
                ("depress", "depression"),
                ("alzheim", "alzheimer"),
                ("parkins", "parkinson"),
                ("anxiet", "anxiety"),
                ("schizo", "schizophrenia")
            ]

            for root, full_term in common_roots:
                if (root in query_term and root in paper_term) or \
                        (full_term in query_term and root in paper_term) or \
                        (root in query_term and full_term in paper_term):
                    return True

        return False

    def _get_matched_concepts(self, query_concepts: dict, graph_topics: dict) -> dict:
        """获取匹配的概念详情"""

        matched = {}

        for category, concepts in query_concepts.items():
            matched[category] = []
            for concept in concepts:
                if concept.lower() in graph_topics:
                    matched[category].append({
                        "concept": concept,
                        "frequency": graph_topics[concept.lower()]
                    })

        return matched

    async def _community_based_query(self, query: str) -> dict:
        """
        基于社区的查询 - 改进版

        流程：
        1. 提取查询关键词
        2. 匹配相关研究社区
        3. 从社区中提取候选论文
        4. 调用 comprehensive_query 进行完整过滤和评分
        """

        logger.info(f"社区驱动查询: {query}")

        # 1. 提取查询关键词
        query_keywords_data = await self._extract_keywords_from_query(query)
        query_keywords = query_keywords_data.get("include", [])
        must_have_keywords = query_keywords_data.get("must_have", [])
        exclude_keywords = query_keywords_data.get("exclude", [])
        year_range = query_keywords_data.get("year_range", {"start": None, "end": None})

        logger.info(f"提取的查询关键词: {query_keywords}")
        logger.info(f"必须关键词: {must_have_keywords}")
        logger.info(f"排除关键词: {exclude_keywords}")
        logger.info(f"年份范围: {year_range}")

        # 2. 获取所有社区
        communities = await self.communities_storage.get_by_id("communities")
        if not communities:
            logger.warning("未找到社区信息，回退到综合查询")
            return await self._comprehensive_query(query)

        logger.info(f"总共有 {len(communities)} 个社区")

        # 3. 计算查询与每个社区的相关性
        community_scores = []
        for community_id, community_info in communities.items():
            community_papers = list(community_info.get("nodes", []))

            if not community_papers:
                continue

            relevance_score = await self._calculate_community_relevance_by_keywords(
                query_keywords,
                community_papers
            )

            if relevance_score > 0.7:  # 保留所有有相关性的社区
                community_scores.append({
                    "community_id": community_id,
                    "level": community_info.get("level", 0),
                    "num_papers": len(community_papers),
                    "relevance_score": relevance_score,
                    "papers": community_papers
                })
                logger.info(f"社区 {community_id}: {len(community_papers)} 篇论文, 相关性: {relevance_score:.3f}")

        if not community_scores:
            logger.warning("未找到相关社区，回退到综合查询")
            return await self._comprehensive_query(query)

        # 4. 从社区中提取候选论文ID
        candidate_paper_ids = set()
        for community in community_scores:
            candidate_paper_ids.update(community["papers"])

        logger.info(f"从 {len(community_scores)} 个社区中提取了 {len(candidate_paper_ids)} 篇候选论文")

        # 5. 调用 comprehensive_query，但限制在候选论文范围内
        comprehensive_results = await self._comprehensive_query_with_candidates(
            query,
            candidate_paper_ids
        )

        # 6. 添加社区信息到结果中
        comprehensive_results["mode"] = "community_based"
        comprehensive_results["num_communities"] = len(community_scores)
        comprehensive_results["top_communities"] = [
            {
                "id": c["community_id"],
                "relevance": c["relevance_score"],
                "num_papers": c["num_papers"]
            }
            for c in community_scores
        ]

        return comprehensive_results

    async def _comprehensive_query_with_candidates(
            self,
            query: str,
            candidate_paper_ids: set = None
    ) -> dict:
        """
        综合查询 - 支持限制候选论文范围

        Args:
            query: 查询字符串
            candidate_paper_ids: 候选论文ID集合（如果为None，则搜索全部）
        """

        # 0. 预先验证查询相关性
        relevance_check = await self._check_query_relevance(query)
        if not relevance_check["is_relevant"]:
            return {
                "query": query,
                "mode": "comprehensive",
                "total_papers": 0,
                "papers": [],
                "relevance_warning": relevance_check["reason"],
                "suggested_topics": relevance_check.get("available_topics", [])
            }

        # 1. 关键词查询
        keyword_results = await self._keyword_query(query, candidate_paper_ids)

        # 2. 语义查询
        semantic_results = await self._semantic_query(query, candidate_paper_ids)

        # 3. 结构化查询
        structural_results = await self._structural_query(query, candidate_paper_ids)

        # 获取必须匹配关键词、排除关键词和年份范围
        must_have_keywords = keyword_results.get("must_have_keywords", [])
        exclude_keywords = keyword_results.get("exclude_keywords", [])
        year_range = keyword_results.get("year_range", {"start": None, "end": None})

        # 4. 结果融合（完全复用 comprehensive_query 的逻辑）
        all_papers = {}

        # 4.1 处理关键词结果
        for paper in keyword_results["papers"]:
            paper_id = paper["paper_id"]
            if await self._verify_paper_relevance(paper_id, query):
                all_papers[paper_id] = {
                    **paper,
                    "keyword_score": paper["keyword_score"],
                    "semantic_score": 0.0,
                    "structural_score": 0.0,
                    "has_keyword_match": True
                }

        # 4.2 处理语义结果
        for paper in semantic_results["papers"]:
            paper_id = paper["paper_id"]
            if await self._verify_paper_relevance(paper_id, query):
                if paper_id in all_papers:
                    all_papers[paper_id]["semantic_score"] = paper["similarity_score"]
                else:
                    all_papers[paper_id] = {
                        **paper,
                        "keyword_score": 0.0,
                        "semantic_score": paper["similarity_score"],
                        "structural_score": 0.0,
                        "has_keyword_match": False
                    }

        # 4.3 处理结构化结果
        for paper in structural_results.get("papers", []):
            paper_id = paper["paper_id"]
            if await self._verify_paper_relevance(paper_id, query):
                if paper_id in all_papers:
                    all_papers[paper_id]["structural_score"] = paper["structural_score"]
                else:
                    all_papers[paper_id] = {
                        **paper,
                        "keyword_score": 0.0,
                        "semantic_score": 0.0,
                        "structural_score": paper["structural_score"],
                        "has_keyword_match": False
                    }

        # 4.4 应用必须匹配关键词过滤
        if must_have_keywords:
            all_paper_ids = list(all_papers.keys())
            filtered_paper_ids = await self._filter_must_have_keywords(all_paper_ids, must_have_keywords)
            all_papers = {pid: all_papers[pid] for pid in filtered_paper_ids}
            logger.info(f"综合查询必须关键词过滤: {len(all_paper_ids)} -> {len(all_papers)} 篇论文")

        # 4.5 应用年份范围过滤
        if year_range.get("start") or year_range.get("end"):
            all_paper_ids = list(all_papers.keys())
            filtered_paper_ids = await self._filter_by_year_range(all_paper_ids, year_range)
            all_papers = {pid: all_papers[pid] for pid in filtered_paper_ids}
            logger.info(f"综合查询年份过滤: {len(all_paper_ids)} -> {len(all_papers)} 篇论文")

        # 4.6 应用排除关键词过滤
        if exclude_keywords:
            all_paper_ids = list(all_papers.keys())
            filtered_paper_ids = await self._filter_out_excluded_papers(all_paper_ids, exclude_keywords)
            all_papers = {pid: all_papers[pid] for pid in filtered_paper_ids}
            logger.info(f"综合查询排除关键词过滤: {len(all_paper_ids)} -> {len(all_papers)} 篇论文")

        # 5. 计算综合得分
        for paper_id, paper in all_papers.items():
            base_score = (
                    paper["keyword_score"] * 0.25 +
                    paper["semantic_score"] * 0.4 +
                    paper["structural_score"] * 0.35
            )
            keyword_bonus = 0.1 if paper.get("has_keyword_match", False) else 0.0
            paper["comprehensive_score"] = min(base_score + keyword_bonus, 1.0)

        # 6. 最终过滤
        filtered_papers = {
            pid: paper for pid, paper in all_papers.items()
            if paper["comprehensive_score"] > 0.5 and paper["keyword_score"] > 0
        }

        # 7. 排序
        sorted_papers = sorted(
            filtered_papers.values(),
            key=lambda x: x["comprehensive_score"],
            reverse=True
        )

        return {
            "query": query,
            "mode": "comprehensive",
            "query_keywords": keyword_results.get("query_keywords"),
            "must_have_keywords": must_have_keywords,
            "exclude_keywords": exclude_keywords,
            "year_range": year_range,
            "total_papers": len(sorted_papers),
            "papers": sorted_papers,
            "keyword_matches": keyword_results.get("keyword_matches", {}),
            "semantic_coverage": len(semantic_results["papers"]),
            "structural_coverage": len(structural_results.get("papers", [])),
            "filters_applied": {
                "must_have_keywords": len(must_have_keywords) > 0,
                "year_range": year_range.get("start") or year_range.get("end"),
                "exclude_keywords": len(exclude_keywords) > 0
            }
        }

    async def _calculate_community_relevance_by_keywords(
            self,
            query_keywords: List[str],
            community_papers: List[str]
    ) -> float:
        """
        通过关键词匹配计算社区相关性（改进版）

        Args:
            query_keywords: 查询关键词列表
            community_papers: 社区内的论文ID列表

        Returns:
            相关性得分 (0-1)
        """

        if not query_keywords or not community_papers:
            return 0.0

        # 获取社区内所有论文的关键词
        paper_nodes = await self.paper_nodes_storage.get_by_ids(community_papers)

        # 聚合社区内所有关键词
        community_all_keywords = set()
        for node in paper_nodes:
            if not node:
                continue

            keywords = node.get("keywords", {})
            for kw_list in keywords.values():
                if kw_list:
                    community_all_keywords.update([kw.lower() for kw in kw_list])

        if not community_all_keywords:
            return 0.0

        # 计算查询关键词的匹配度
        matched_count = 0
        for query_kw in query_keywords:
            query_kw_lower = query_kw.lower()

            # 1. 精确匹配
            if query_kw_lower in community_all_keywords:
                matched_count += 1
                continue

            # 2. 使用映射表匹配
            mapped_keywords = self._get_mapped_keywords(query_kw_lower)
            if any(mk in community_all_keywords for mk in mapped_keywords):
                matched_count += 1
                continue

            # 3. 部分匹配
            for comm_kw in community_all_keywords:
                if query_kw_lower in comm_kw or comm_kw in query_kw_lower:
                    matched_count += 0.5  # 部分匹配权重降低
                    break

        # 计算相关性得分
        relevance = matched_count / len(query_keywords)

        logger.debug(f"社区关键词匹配: {matched_count}/{len(query_keywords)} = {relevance:.3f}")

        return relevance

    def _calculate_structural_similarity_with_query(self, query: str, paper_node: dict) -> float:
        """
        计算查询与论文的结构化特征相似度

        Args:
            query: 查询字符串
            paper_node: 论文节点数据

        Returns:
            结构化相似度分数 (0-1)
        """

        # 从查询中提取结构化信息（简化版）
        query_lower = query.lower()

        similarity_score = 0.0
        total_weight = 0.0

        # 1. 检查干预类型匹配
        intervention_keywords = ['rtms', 'tms', 'tdcs', 'ect', 'stimulation']
        paper_intervention = paper_node.get("structured_features", {}).get("intervention", {})
        intervention_type = paper_intervention.get("type", "").lower()

        if intervention_type:
            total_weight += 0.4
            for keyword in intervention_keywords:
                if keyword in query_lower and keyword in intervention_type:
                    similarity_score += 0.4
                    break

        # 2. 检查疾病/人群匹配
        disease_keywords = ['depression', 'anxiety', 'schizophrenia', 'alzheimer', 'parkinson']
        paper_participants = paper_node.get("structured_features", {}).get("participants", {})
        condition = paper_participants.get("condition", "").lower()

        if condition:
            total_weight += 0.3
            for keyword in disease_keywords:
                if keyword in query_lower and keyword in condition:
                    similarity_score += 0.3
                    break

        # 3. 检查研究设计匹配
        design_keywords = ['rct', 'randomized', 'controlled', 'trial', 'crossover']
        paper_design = paper_node.get("structured_features", {}).get("design", {})
        design_type = paper_design.get("type", "").lower()

        if design_type:
            total_weight += 0.3
            for keyword in design_keywords:
                if keyword in query_lower and keyword in design_type:
                    similarity_score += 0.3
                    break

        # 归一化
        if total_weight > 0:
            return similarity_score / total_weight
        else:
            return 0.0


    async def _calculate_semantic_similarity_with_query(self, query: str, paper_node: dict) -> float:
        """
        计算查询与论文的语义相似度

        Args:
            query: 查询字符串
            paper_node: 论文节点数据

        Returns:
            语义相似度分数 (0-1)
        """

        # 1. 获取查询的embedding
        query_embedding = await self._compute_text_embedding(query)

        if not query_embedding:
            return 0.0

        # 2. 获取论文的embedding（优先使用缓存的）
        paper_id = paper_node.get("id") or paper_node.get("paper_id")

        # 尝试从语义索引获取
        semantic_data = await self.semantic_index.get_by_id(paper_id)

        if semantic_data and "embedding" in semantic_data:
            paper_embedding = semantic_data["embedding"]
        elif "abstract_embedding" in paper_node and paper_node["abstract_embedding"]:
            paper_embedding = paper_node["abstract_embedding"]
        else:
            # 如果没有缓存，现场计算
            abstract = paper_node.get("abstract", "")
            title = paper_node.get("title", "")
            combined_text = f"{title}. {abstract}"

            if not combined_text.strip():
                return 0.0

            paper_embedding = await self._compute_text_embedding(combined_text)

            if not paper_embedding:
                return 0.0

        # 3. 计算余弦相似度
        similarity = self._calculate_semantic_similarity(query_embedding, paper_embedding)

        return similarity

    async def _keyword_query(self, query: str, candidate_paper_ids: set = None) -> dict:
        """关键词查询 - 支持必须匹配关键词、排除关键词和年份范围"""

        # 提取包含关键词、必须匹配关键词、排除关键词和年份范围
        keyword_extraction = await self._extract_keywords_from_query(query)
        query_keywords = keyword_extraction["include"]
        must_have_keywords = keyword_extraction["must_have"]
        exclude_keywords = keyword_extraction["exclude"]
        year_range = keyword_extraction["year_range"]

        logger.info(f"查询关键词: {query_keywords}")
        if must_have_keywords:
            logger.info(f"必须匹配关键词: {must_have_keywords}")
        if exclude_keywords:
            logger.info(f"排除关键词: {exclude_keywords}")
        if year_range.get("start") or year_range.get("end"):
            logger.info(f"年份范围: {year_range}")

        # 获取关键词索引
        keyword_index = await self._load_keyword_index()

        if not keyword_index:
            logger.warning("关键词索引为空")
            return {
                "query": query,
                "mode": "keyword",
                "query_keywords": query_keywords,
                "must_have_keywords": must_have_keywords,
                "exclude_keywords": exclude_keywords,
                "year_range": year_range,
                "total_papers": 0,
                "papers": []
            }

        # 查找候选论文
        matched_papers = await self._find_papers_by_keywords(query_keywords)

        # 如果指定了候选范围，只保留在范围内的论文
        if candidate_paper_ids is not None:
            matched_papers = [pid for pid in matched_papers if pid in candidate_paper_ids]
            logger.info(f"限制在候选范围后: {len(matched_papers)} 篇论文")

        logger.info(f"关键词匹配找到 {len(matched_papers)} 篇候选论文")

        # 1. 必须匹配关键词过滤
        if must_have_keywords:
            matched_papers = await self._filter_must_have_keywords(matched_papers, must_have_keywords)
            logger.info(f"必须关键词过滤后剩余论文: {len(matched_papers)}")

        # 2. 年份范围过滤
        if year_range.get("start") or year_range.get("end"):
            matched_papers = await self._filter_by_year_range(matched_papers, year_range)
            logger.info(f"年份过滤后剩余论文: {len(matched_papers)}")

        # 3. 排除关键词过滤
        if exclude_keywords:
            matched_papers = await self._filter_out_excluded_papers(matched_papers, exclude_keywords)
            logger.info(f"排除关键词过滤后剩余论文: {len(matched_papers)}")

        # 评分
        scored_papers = await self._score_papers_by_keywords(matched_papers, query_keywords)

        return {
            "query": query,
            "mode": "keyword",
            "query_keywords": query_keywords,
            "must_have_keywords": must_have_keywords,
            "exclude_keywords": exclude_keywords,
            "year_range": year_range,
            "total_papers": len(scored_papers),
            "papers": scored_papers,
            "keyword_matches": await self._get_keyword_match_details(query_keywords)
        }

    async def _semantic_query(self, query: str, candidate_paper_ids: set = None) -> dict:
        """语义查询"""

        # 计算查询的embedding
        query_embedding = await self._compute_text_embedding(query)

        if not query_embedding:
            return {"query": query, "mode": "semantic", "papers": []}

        # 获取所有论文的embedding
        semantic_keys = await self.semantic_index.all_keys()

        # 如果指定了候选范围，只处理候选论文
        if candidate_paper_ids is not None:
            semantic_keys = [key for key in semantic_keys if key in candidate_paper_ids]
            logger.info(f"语义查询限制在 {len(semantic_keys)} 篇候选论文范围内")

        semantic_data = {}

        if semantic_keys:
            semantic_values = await self.semantic_index.get_by_ids(semantic_keys)
            semantic_data = {key: value for key, value in zip(semantic_keys, semantic_values) if value is not None}

        # 获取所有论文节点
        paper_keys = list(semantic_data.keys())
        all_papers = {}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            all_papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

        # 计算相似度
        similarities = []
        for paper_id, semantic_info in semantic_data.items():
            if paper_id in all_papers:
                similarity = self._calculate_semantic_similarity(
                    query_embedding, semantic_info["embedding"]
                )
                if similarity > self.semantic_similarity_threshold:
                    similarities.append({
                        "paper_id": paper_id,
                        "title": all_papers[paper_id]["title"],
                        "similarity_score": similarity
                    })

        # 排序
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {
            "query": query,
            "mode": "semantic",
            "total_papers": len(similarities),
            "papers": similarities
        }

    async def _load_keyword_index(self) -> dict:
        """加载关键词索引"""

        keyword_keys = await self.keyword_index.all_keys()
        keyword_index = {}

        if keyword_keys:
            keyword_values = await self.keyword_index.get_by_ids(keyword_keys)
            keyword_index = {key: value for key, value in zip(keyword_keys, keyword_values) if value is not None}

        return keyword_index

    async def _separate_abstract_and_fulltext(self, paper_id: str) -> Tuple[str, str]:
        """分离摘要和全文内容，利用meta_analysis_graphrag的资源"""

        try:
            # 1. 从raw_papers_storage获取完整论文内容
            full_paper_content = await self.meta_graphrag.raw_papers_storage.get_by_id(paper_id)
            if not full_paper_content:
                logger.warning(f"未找到论文 {paper_id} 的完整内容")
                return "", ""

            paper_content = full_paper_content.get("content", "")

            # 2. 从raw_papers_chunks获取已处理的chunks
            all_chunk_keys = await self.meta_graphrag.raw_papers_chunks.all_keys()
            all_chunks = {}

            if all_chunk_keys:
                chunk_values = await self.meta_graphrag.raw_papers_chunks.get_by_ids(all_chunk_keys)
                all_chunks = {key: value for key, value in zip(all_chunk_keys, chunk_values) if value is not None}

            paper_chunks = [chunk for chunk in all_chunks.values()
                           if chunk.get("full_doc_id") == paper_id]


            # 3. 提取摘要 - 使用meta_analysis_graphrag的方法
            abstract_content = await self._extract_abstract_from_chunks(paper_chunks)

            # 4. 生成全文摘要 - 对全文进行智能总结
            fulltext_summary = await self._generate_fulltext_summary(paper_chunks)

            return abstract_content, fulltext_summary

        except Exception as e:
            logger.error(f"从存储中提取摘要和全文失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "", ""

    async def _extract_abstract_from_chunks(self, paper_chunks: List[dict]) -> str:
        """从chunks中提取摘要，复用meta_analysis_graphrag的逻辑"""

        # 使用meta_analysis_graphrag的方法查找abstract chunks
        abstract_chunks = self.meta_graphrag._find_chunks_by_type(paper_chunks, "abstract")

        if abstract_chunks:
            # 如果找到多个abstract chunks，选择最相关的
            best_abstract = self._select_best_abstract_chunk(abstract_chunks)
            return best_abstract["content"].strip()

        # 如果没有找到abstract chunks，查找包含摘要关键词的chunks
        for chunk in paper_chunks:
            content_lower = chunk["content"].lower()
            if any(keyword in content_lower for keyword in
                   ["abstract", "摘要", "summary", "a b s t r a c t"]):
                # 提取摘要部分
                abstract_text = self._extract_abstract_from_chunk_content(chunk["content"])
                if abstract_text:
                    return abstract_text

        sorted_chunks = sorted(paper_chunks, key=lambda x: x.get("chunk_order_index", 0))
        first_chunks = sorted_chunks[:3]
        abstract_parts = []
        for chunk in first_chunks:
            content = chunk["content"].strip()
            if len(content) > 99:
                abstract_parts.append(content)

        if abstract_parts:
            combined_abstract = " ".join(abstract_parts)
            return combined_abstract

        return ""

    def _select_best_abstract_chunk(self, abstract_chunks: List[dict]) -> dict:
        """选择最佳的摘要chunk"""

        if len(abstract_chunks) == 1:
            return abstract_chunks[0]

        # 评分标准：包含更多摘要关键词的chunk得分更高
        abstract_keywords = [
            "objective", "method", "result", "conclusion",
            "目的", "方法", "结果", "结论",
            "background", "aim", "finding", "implication"
        ]

        best_chunk = abstract_chunks[0]
        best_score = 0

        for chunk in abstract_chunks:
            content_lower = chunk["content"].lower()
            score = sum(1 for keyword in abstract_keywords if keyword in content_lower)

            # 长度适中的chunk加分
            content_length = len(chunk["content"])
            if 1000 <= content_length <= 4000:
                score += 2
            elif content_length > 4000:
                score -= 1

            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk

    def _extract_abstract_from_chunk_content(self, content: str) -> str:
        """从chunk内容中提取摘要文本"""

        # 摘要提取模式
        abstract_patterns = [
            r'(?i)abstract\s*[:：]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|背景|关键词|引言|1\.|method))',
            r'(?i)摘\s*要\s*[:：]?\s*(.*?)(?=\n\s*(?:关键词|keywords?|引言|introduction|方法|method))',
            r'(?i)summary\s*[:：]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|method))'
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                abstract_text = match.group(1).strip()
                # 清理文本
                abstract_text = re.sub(r'\n+', ' ', abstract_text)
                abstract_text = re.sub(r'\s+', ' ', abstract_text)

                # 长度检查
                if 50 <= len(abstract_text) <= 2000:
                    return abstract_text

        return ""

    async def _generate_fulltext_summary(self, paper_chunks: List[dict]) -> str:
        """生成全文的结构化摘要"""

        # 1. 按类型分类chunks
        methods_chunks = self.meta_graphrag._find_chunks_by_type(paper_chunks, "methods")
        results_chunks = self.meta_graphrag._find_chunks_by_type(paper_chunks, "results")
        discussion_chunks = self.meta_graphrag._find_chunks_by_type(paper_chunks, "discussion")

        # 2. 选择关键chunks进行总结
        key_chunks = []

        # 选择最重要的methods chunk
        if methods_chunks:
            best_methods = self._select_chunk_by_keywords(
                methods_chunks,
                ["participants", "procedure", "intervention", "design", "randomized"]
            )
            if best_methods:
                key_chunks.append(("Methods", best_methods["content"]))

        # 选择最重要的results chunk
        if results_chunks:
            best_results = self._select_chunk_by_keywords(
                results_chunks,
                ["significant", "effect", "improvement", "baseline", "outcome"]
            )
            if best_results:
                key_chunks.append(("Results", best_results["content"]))

        # 选择discussion chunk（如果有的话）
        if discussion_chunks:
            best_discussion = self._select_chunk_by_keywords(
                discussion_chunks,
                ["conclusion", "implication", "limitation", "future"]
            )
            if best_discussion:
                key_chunks.append(("Discussion", best_discussion["content"]))

        # 3. 如果关键chunks不足，补充其他重要chunks
            remaining_chunks = [chunk for chunk in paper_chunks
                              if chunk not in [kc[1] for kc in key_chunks]]

            # 按chunk质量排序（包含更多关键词的chunk优先）
            scored_chunks = []
            for chunk in remaining_chunks:
                score = self._score_chunk_importance(chunk["content"])
                scored_chunks.append((score, chunk))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            # 补充到3个chunks
            for score, chunk in scored_chunks[:3-len(key_chunks)]:
                key_chunks.append(("Content", chunk["content"]))

        # 4. 使用LLM生成结构化摘要
        if key_chunks:
            return await self._llm_generate_fulltext_summary(key_chunks)
        else:
            # 备选方案：直接拼接前几个chunks
            combined_text = "\n\n".join([chunk["content"] for chunk in paper_chunks[:3]])
            return combined_text

    def _select_chunk_by_keywords(self, chunks: List[dict], keywords: List[str]) -> dict:
        """根据关键词选择最相关的chunk"""

        if not chunks:
            return None

        best_chunk = None
        best_score = 0

        for chunk in chunks:
            content_lower = chunk["content"].lower()
            score = sum(1 for keyword in keywords if keyword in content_lower)

            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk if best_score > 0 else chunks[0]

    def _score_chunk_importance(self, content: str) -> float:
        """评估chunk的重要性"""

        stats_score = 0

        content_lower = content.lower()

        # 重要关键词
        important_keywords = [
            "participants", "subjects", "patients", "intervention", "treatment",
            "randomized", "controlled", "trial", "outcome", "results", "significant",
            "effect", "improvement", "baseline", "follow-up", "analysis", "conclusion"
        ]

        # 计算关键词密度
        keyword_count = sum(1 for keyword in important_keywords if keyword in content_lower)

        # 长度适中加分
        length_score = 0
        if 200 <= len(content) <= 1000:
            length_score = 1

        # 包含数字和统计信息加分
        if re.search(r'\d+\.\d+|p\s*[<>=]\s*\d+\.\d+|95%\s*ci', content_lower):
            stats_score = 1

        return keyword_count + length_score + stats_score

    async def _llm_generate_fulltext_summary(self, key_chunks: List[Tuple[str, str]]) -> str:
        """使用LLM生成全文摘要"""

        # 构建prompt
        chunks_text = ""
        for section_type, content in key_chunks:
            chunks_text += f"\n\n[{section_type}]\n{content}"

        summary_prompt = PROMPTS["summary_prompt"].format(chunks_text=chunks_text)

        try:
            response = await self.meta_graphrag.cheap_model_func(summary_prompt)

            # 清理和验证摘要
            summary = response.strip()

            return summary

        except Exception as e:
            logger.error(f"LLM生成全文摘要失败: {e}")
            return self._create_fallback_summary(key_chunks)


    def _get_default_structured_features(self) -> dict:
        """获取默认结构化特征"""

        return {
            "participants": {
                "condition": "",
                "sample_size": 0,
                "age_range": "",
                "gender_ratio": ""
            },
            "intervention": {
                "type": "",
                "frequency": "",
                "intensity": "",
                "target": "",
                "sessions": 0,
                "duration": ""
            },
            "outcomes": {
                "primary": [],
                "secondary": [],
                "measures": []
            },
            "design": {
                "study_type": "",
                "blinding": "",
                "randomization": ""
            }
        }

    async def _extract_paper_id(self, paper_content: str) -> str:
        """提取论文ID - 从meta_graphrag的存储中获取"""

        # meta_graphrag.ainsert_papers已经处理了DOI提取和hash生成
        # 直接从raw_papers_storage获取最新的paper_id
        all_paper_ids = await self.meta_graphrag.raw_papers_storage.all_keys()

        if not all_paper_ids:
            logger.error("无法从raw_papers_storage获取paper_id")
            # 最后的fallback
            return compute_mdhash_id(paper_content.strip(), prefix="paper-")

        # 返回最新插入的paper_id（可能是DOI或hash）
        paper_id = all_paper_ids[-1]
        logger.info(f"提取到论文ID: {paper_id}")
        return paper_id

    async def _extract_keywords_from_query(self, query: str) -> dict:
        """从查询中提取关键词、必须匹配关键词、排除关键词和年份范围 - 使用LLM，支持PICO分类"""

        query_prompt = PROMPTS["query_prompt"].format(query=query)

        try:
            response = await self.meta_graphrag.cheap_model_func(query_prompt)

            # 解析JSON响应
            import json
            from datetime import datetime

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                # 提取包含关键词（扁平化PICO分类）
                include_keywords_dict = result.get("include_keywords", {})
                include_keywords = []

                for category, keywords in include_keywords_dict.items():
                    for kw in keywords:
                        kw_clean = kw.lower().strip()
                        if kw_clean and kw_clean not in include_keywords:
                            include_keywords.append(kw_clean)

                # 提取必须匹配关键词
                must_have_keywords = [kw.lower().strip() for kw in result.get("must_have_keywords", [])]

                # 提取排除关键词
                exclude_keywords = [kw.lower().strip() for kw in result.get("exclude_keywords", [])]

                # 提取年份范围
                year_range = result.get("year_range", {"start": None, "end": None})

                logger.info(f"LLM提取到包含关键词 ({len(include_keywords)}): {include_keywords}")
                if must_have_keywords:
                    logger.info(f"LLM提取到必须匹配关键词 ({len(must_have_keywords)}): {must_have_keywords}")
                if exclude_keywords:
                    logger.info(f"LLM提取到排除关键词 ({len(exclude_keywords)}): {exclude_keywords}")
                if year_range.get("start") or year_range.get("end"):
                    logger.info(f"LLM提取到年份范围: {year_range}")

                return {
                    "include": include_keywords,
                    "must_have": must_have_keywords,
                    "exclude": exclude_keywords,
                    "year_range": year_range,
                    "include_by_category": include_keywords_dict  # 保留分类信息供调试
                }
            else:
                raise ValueError("LLM未返回有效的JSON格式")

        except Exception as e:
            logger.error(f"LLM关键词提取失败: {e}")
            logger.warning("回退到简单分词方法")

            # 回退方案：简单分词
            return {
                "include": query.lower().split(),
                "must_have": [],
                "exclude": [],
                "year_range": {"start": None, "end": None},
                "include_by_category": {}
            }

    async def _filter_must_have_keywords(self, paper_ids: List[str], must_have_keywords: List[str]) -> List[str]:
        """过滤出必须包含指定关键词的论文"""

        if not must_have_keywords:
            return paper_ids

        filtered_papers = []

        # 获取所有论文节点
        paper_keys = await self.paper_nodes_storage.all_keys()
        all_papers = {}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            all_papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

        # 扩展必须匹配关键词（使用映射）
        expanded_must_have_keywords = {}
        for must_kw in must_have_keywords:
            expanded_variants = set([must_kw.lower()])
            # 获取映射的同义词
            mapped = self._get_mapped_keywords(must_kw)
            expanded_variants.update([m.lower() for m in mapped])
            expanded_must_have_keywords[must_kw] = expanded_variants

        logger.info(f"必须匹配关键词及其变体: {expanded_must_have_keywords}")

        for paper_id in paper_ids:
            if paper_id not in all_papers:
                continue

            paper_node = all_papers[paper_id]
            paper_keywords = paper_node.get("keywords", {})

            # 将论文的所有关键词展平
            all_paper_keywords_lower = set()
            for category, keywords in paper_keywords.items():
                for kw in keywords:
                    all_paper_keywords_lower.add(kw.lower())

            # 检查是否所有必须关键词都匹配
            all_must_keywords_matched = True
            matched_must_keywords = []
            unmatched_must_keywords = []

            for must_kw, variants in expanded_must_have_keywords.items():
                # 检查是否有任何变体匹配
                has_match = False
                matched_variant = None

                for variant in variants:
                    # 精确匹配
                    if variant in all_paper_keywords_lower:
                        has_match = True
                        matched_variant = variant
                        break

                    # 部分匹配（变体是论文关键词的子串，或反之）
                    for paper_kw in all_paper_keywords_lower:
                        if variant in paper_kw or paper_kw in variant:
                            # 确保是有意义的匹配（避免过短的子串）
                            if len(variant) >= 4 or len(paper_kw) >= 4:
                                has_match = True
                                matched_variant = paper_kw
                                break

                    if has_match:
                        break

                if has_match:
                    matched_must_keywords.append(f"{must_kw} (matched: {matched_variant})")
                else:
                    all_must_keywords_matched = False
                    unmatched_must_keywords.append(must_kw)

            if all_must_keywords_matched:
                logger.info(f"论文 {paper_id[:30]}... 匹配所有必须关键词: {matched_must_keywords}")
                filtered_papers.append(paper_id)
            else:
                logger.info(f"论文 {paper_id[:30]}... 缺少必须关键词: {unmatched_must_keywords}")

        logger.info(f"必须关键词过滤: {len(paper_ids)} -> {len(filtered_papers)} 篇论文")

        return filtered_papers


    async def _filter_by_year_range(self, paper_ids: List[str], year_range: dict) -> List[str]:
        """根据年份范围过滤论文"""

        start_year = year_range.get("start")
        end_year = year_range.get("end")

        # 如果没有年份限制，直接返回
        if start_year is None and end_year is None:
            return paper_ids

        filtered_papers = []

        # 获取所有论文的元数据
        all_paper_ids = await self.meta_graphrag.evaluated_papers_storage.all_keys()
        all_papers_data = {}

        if all_paper_ids:
            paper_values = await self.meta_graphrag.evaluated_papers_storage.get_by_ids(all_paper_ids)
            all_papers_data = {key: value for key, value in zip(all_paper_ids, paper_values) if value is not None}

        for paper_id in paper_ids:
            if paper_id not in all_papers_data:
                continue

            paper_data = all_papers_data[paper_id]
            paper_year = paper_data.get("year")

            # 如果论文没有年份信息，跳过
            if paper_year is None:
                logger.debug(f"论文 {paper_id[:30]}... 没有年份信息，跳过")
                continue

            # 检查年份范围
            if start_year is not None and paper_year < start_year:
                logger.debug(f"论文 {paper_id[:30]}... 年份 {paper_year} < {start_year}，排除")
                continue

            if end_year is not None and paper_year > end_year:
                logger.debug(f"论文 {paper_id[:30]}... 年份 {paper_year} > {end_year}，排除")
                continue

            filtered_papers.append(paper_id)

        if start_year or end_year:
            year_desc = f"{start_year or '∞'} - {end_year or '∞'}"
            logger.info(f"年份范围过滤 ({year_desc}): {len(paper_ids)} -> {len(filtered_papers)} 篇论文")

        return filtered_papers

    async def _filter_out_excluded_papers(self, paper_ids: List[str], exclude_keywords: List[str]) -> List[str]:
        """过滤掉包含排除关键词的论文"""

        if not exclude_keywords:
            return paper_ids

        filtered_papers = []

        # 获取所有论文节点
        paper_keys = await self.paper_nodes_storage.all_keys()
        all_papers = {}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            all_papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

        # 扩展排除关键词（使用映射）
        expanded_exclude_keywords = set()
        for exclude_kw in exclude_keywords:
            expanded_exclude_keywords.add(exclude_kw.lower())
            # 获取映射的同义词
            mapped = self._get_mapped_keywords(exclude_kw)
            expanded_exclude_keywords.update([m.lower() for m in mapped])

        logger.info(f"扩展后的排除关键词: {expanded_exclude_keywords}")

        for paper_id in paper_ids:
            if paper_id not in all_papers:
                continue

            paper_node = all_papers[paper_id]
            paper_keywords = paper_node.get("keywords", {})

            # 检查论文的所有关键词类别
            should_exclude = False
            matched_exclude_kw = []

            for category, keywords in paper_keywords.items():
                for kw in keywords:
                    kw_lower = kw.lower()
                    # 检查是否匹配任何排除关键词
                    for exclude_kw in expanded_exclude_keywords:
                        if exclude_kw == kw_lower:
                            should_exclude = True
                            matched_exclude_kw.append(kw)
                            break
                        elif re.search(r'\b' + re.escape(exclude_kw) + r'\b', kw_lower):
                            should_exclude = True
                            matched_exclude_kw.append(kw)
                            break
                    if should_exclude:
                        break
                if should_exclude:
                    break

            if should_exclude:
                logger.info(f"排除论文 {paper_id[:30]}... (匹配排除关键词: {matched_exclude_kw})")
            else:
                filtered_papers.append(paper_id)

        logger.info(f"排除关键词过滤: {len(paper_ids)} -> {len(filtered_papers)} 篇论文")

        return filtered_papers


    async def _find_papers_by_keywords(self, query_keywords: List[str]) -> List[str]:
        """基于关键词匹配找到候选论文 - 使用映射表提高匹配准确性，特定TMS类型严格过滤"""

        candidate_papers = set()

        # 获取关键词索引
        keyword_keys = await self.keyword_index.all_keys()
        keyword_index = {}

        if keyword_keys:
            keyword_values = await self.keyword_index.get_by_ids(keyword_keys)
            keyword_index = {key: value for key, value in zip(keyword_keys, keyword_values) if value is not None}

        # 检测查询中的所有特定TMS类型（可能有多个）
        query_tms_types = self._detect_all_query_tms_types(query_keywords)

        # 分离TMS关键词和非TMS关键词
        tms_keywords = []
        non_tms_keywords = []

        for keyword in query_keywords:
            if self._is_specific_tms_type(keyword) or keyword.lower() in ["tms", "transcranial magnetic stimulation"] or "stimulation" in keyword.lower():
                tms_keywords.append(keyword)
            else:
                non_tms_keywords.append(keyword)

        # 如果有特定TMS类型，先通过TMS类型筛选论文池
        if query_tms_types:
            # 获取所有论文
            paper_keys = await self.paper_nodes_storage.all_keys()
            all_paper_ids = list(paper_keys) if paper_keys else []

            # 只保留符合TMS类型的论文
            tms_qualified_papers = set(await self._filter_papers_by_tms_types(
                all_paper_ids,
                query_tms_types
            ))

            logger.info(f"TMS类型预筛选: {query_tms_types}, 符合条件的论文: {len(tms_qualified_papers)}")
        else:
            tms_qualified_papers = None  # 没有TMS类型限制

        # 处理非TMS关键词（或所有关键词如果没有TMS限制）
        keywords_to_process = non_tms_keywords if query_tms_types else query_keywords

        for query_keyword in keywords_to_process:
            matched_papers = set()

            # 1. 精确匹配
            if query_keyword in keyword_index:
                matched_papers.update(keyword_index[query_keyword]["papers"])

            # 2. 使用映射表匹配（疾病、干预、脑区）
            mapped_keywords = self._get_mapped_keywords(query_keyword)
            for mapped_kw in mapped_keywords:
                if mapped_kw in keyword_index:
                    matched_papers.update(keyword_index[mapped_kw]["papers"])

            # 3. 模糊匹配
            for stored_keyword, keyword_info in keyword_index.items():
                # 包含关系
                if query_keyword in stored_keyword or stored_keyword in query_keyword:
                    matched_papers.update(keyword_info["papers"])
                # 高相似度匹配
                elif self._calculate_keyword_similarity_simple(query_keyword, stored_keyword) > 0.85:
                    matched_papers.update(keyword_info["papers"])

            if not matched_papers:
                logger.warning(f"查询关键词 '{query_keyword}' 没有找到任何匹配")

            # 如果有TMS类型限制，只保留符合TMS类型的论文
            if tms_qualified_papers is not None:
                matched_papers = matched_papers & tms_qualified_papers

            candidate_papers.update(matched_papers)

        # 最终确保所有论文都符合TMS类型
        if query_tms_types and candidate_papers:
            candidate_papers = set(await self._filter_papers_by_tms_types(
                list(candidate_papers),
                query_tms_types
            ))
            logger.info(f"最终TMS类型验证后剩余论文: {len(candidate_papers)}")

        return list(candidate_papers)


    def _is_tms_substring_mismatch(self, keyword1: str, keyword2: str) -> bool:
        """检查两个关键词是否是TMS类型的子串误匹配"""

        keyword1_lower = keyword1.lower()
        keyword2_lower = keyword2.lower()

        # 获取两个关键词对应的TMS类型
        tms_type1 = None
        tms_type2 = None

        for tms_type, synonyms in INTERVENTION_SYNONYMS.items():
            synonyms_lower = [s.lower() for s in synonyms]
            if keyword1_lower in synonyms_lower or any(syn in keyword1_lower for syn in synonyms_lower):
                tms_type1 = tms_type
            if keyword2_lower in synonyms_lower or any(syn in keyword2_lower for syn in synonyms_lower):
                tms_type2 = tms_type

        # 如果两个都是TMS类型，但类型不同，则是误匹配
        if tms_type1 and tms_type2 and tms_type1 != tms_type2:
            return True

        return False

    def _detect_all_query_tms_types(self, query_keywords: List[str]) -> List[str]:
        """检测查询中的所有TMS类型（可能有多个）"""

        specific_tms_types = [
            "rtms", "dtms", "atms", "itbs", "ctbs", "tbs",
            "quadripulse", "pas", "artms", "aitbs", "actbs", "atbs"
        ]

        detected_types = []

        for keyword in query_keywords:
            keyword_lower = keyword.lower()

            # 检查是否匹配特定TMS类型（精确匹配，避免子串）
            for tms_type in specific_tms_types:
                if tms_type in INTERVENTION_SYNONYMS:
                    synonyms = [s.lower() for s in INTERVENTION_SYNONYMS[tms_type]]
                    # 精确匹配关键词
                    if keyword_lower in synonyms:
                        if tms_type not in detected_types:
                            detected_types.append(tms_type)
                            logger.info(f"检测到TMS类型: {tms_type} (来自关键词: {keyword})")

        logger.info(f"最终检测到的TMS类型: {detected_types}")
        return detected_types

    async def _filter_papers_by_tms_types(self, paper_ids: List[str], required_tms_types: List[str]) -> List[str]:
        """根据TMS类型过滤论文（支持父子类层次化匹配）"""

        filtered_papers = []

        # 获取所有论文节点
        paper_keys = await self.paper_nodes_storage.all_keys()
        all_papers = {}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            all_papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

        # 【关键修改】收集所有要求的TMS类型的同义词 + 子类
        all_required_variants = set()

        for tms_type in required_tms_types:
            # 1. 添加同义词（同级别）
            synonyms = [s.lower() for s in INTERVENTION_SYNONYMS.get(tms_type, [])]
            all_required_variants.update(synonyms)

            # 2. 添加子类（层次化）
            if tms_type in INTERVENTION_HIERARCHY:
                child_methods = [m.lower() for m in INTERVENTION_HIERARCHY[tms_type]]
                all_required_variants.update(child_methods)

                # 递归添加子类的同义词
                for child in INTERVENTION_HIERARCHY[tms_type]:
                    child_lower = child.lower()
                    if child_lower in INTERVENTION_SYNONYMS:
                        child_synonyms = [s.lower() for s in INTERVENTION_SYNONYMS[child_lower]]
                        all_required_variants.update(child_synonyms)

        logger.info(f"要求的TMS类型: {required_tms_types}")
        logger.info(f"扩展后的匹配词（含子类）: {list(all_required_variants)[:20]}")  # 只显示前20个

        for paper_id in paper_ids:
            if paper_id not in all_papers:
                continue

            paper_node = all_papers[paper_id]
            paper_keywords = paper_node.get("keywords", {})

            # 获取论文的干预关键词
            intervention_keywords = paper_keywords.get("intervention", [])

            # 检查是否包含任一匹配的TMS类型（精确匹配，不能是子串）
            has_matching_tms = False
            matched_tms = []

            for intervention_kw in intervention_keywords:
                intervention_kw_lower = intervention_kw.lower()

                # 精确匹配或完整同义词匹配
                for required_variant in all_required_variants:
                    # 精确匹配
                    if intervention_kw_lower == required_variant:
                        has_matching_tms = True
                        matched_tms.append(intervention_kw)
                        break
                    # 完整词匹配（避免子串误匹配）
                    elif required_variant in intervention_kw_lower.split():
                        has_matching_tms = True
                        matched_tms.append(intervention_kw)
                        break

                if has_matching_tms:
                    break

            if has_matching_tms:
                logger.info(f"论文 {paper_id[:30]}... 匹配成功: {matched_tms}")
                filtered_papers.append(paper_id)
            else:
                logger.info(f"论文 {paper_id[:30]}... 不匹配，干预关键词: {intervention_keywords}")

        logger.info(f"TMS类型过滤: {len(paper_ids)} -> {len(filtered_papers)} 篇论文")
        return filtered_papers


    def _get_mapped_keywords(self, keyword: str) -> List[str]:
        """获取关键词的所有映射（同义词、相关词）"""

        keyword_lower = keyword.lower()
        mapped = set()

        # 1. 疾病映射
        for disease, synonyms in DISEASE_CONDITIONS.items():
            if keyword_lower in [s.lower() for s in synonyms]:
                mapped.update([s.lower() for s in synonyms])

        # 2. 干预方法映射（包含TMS类型）
        # 只映射到同一类型的同义词，不跨类型映射
        for intervention, synonyms in INTERVENTION_SYNONYMS.items():
            synonyms_lower = [s.lower() for s in synonyms]
            if keyword_lower in synonyms_lower:
                # 只添加该类型的同义词，不添加其他类型
                mapped.add(intervention)
                mapped.update(synonyms_lower)
                break  # 找到匹配后立即退出，避免跨类型映射

        # 3. 脑区映射
        for region, keywords in BRAIN_REGIONS.items():
            if keyword_lower in [k.lower() for k in keywords]:
                mapped.add(region)
                mapped.update([k.lower() for k in keywords])

        # 4. 结局指标映射
        for outcome, aliases in OUTCOME_ALIASES.items():
            if keyword_lower in [a.lower() for a in aliases]:
                mapped.add(outcome)
                mapped.update([a.lower() for a in aliases])

        # 5. 通用TMS扩展（仅当关键词是通用TMS时）
        if keyword_lower in ["tms", "transcranial magnetic stimulation"]:
            if "tms" in INTERVENTION_HIERARCHY:
                # 通用TMS可以匹配所有TMS子类型
                mapped.update([kw.lower() for kw in INTERVENTION_HIERARCHY["tms"]])

        return list(mapped)

    def _calculate_keyword_similarity_simple(self, keyword1: str, keyword2: str) -> float:
        """
        计算关键词相似性
        保持简单 - 不做复杂的多维度匹配
        """
        # 简单的字符串相似性计算
        if keyword1 == keyword2:
            return 1.0

        # 包含关系
        if keyword1 in keyword2 or keyword2 in keyword1:
            return 0.8

        # 计算公共子串
        common_chars = set(keyword1) & set(keyword2)
        total_chars = set(keyword1) | set(keyword2)

        if not total_chars:
            return 0.0

        return len(common_chars) / len(total_chars)

    async def _score_papers_by_keywords(self, candidate_papers: List[str], query_keywords: List[str]) -> List[dict]:
        """基于关键词为论文评分 - 要求高匹配度"""

        scored_papers = []

        # 获取所有论文节点
        paper_keys = await self.paper_nodes_storage.all_keys()
        all_papers = {}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            all_papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

        logger.info(f"开始为 {len(candidate_papers)} 篇候选论文评分")

        for paper_id in candidate_papers:
            if paper_id not in all_papers:
                logger.warning(f"论文 {paper_id} 不在存储中")
                continue

            paper_node = all_papers[paper_id]

            # 计算关键词匹配得分
            keyword_score, matched_keywords = self._calculate_keyword_match_score(
                query_keywords, paper_node["keywords"]
            )

            logger.info(f"论文 {paper_id[:30]}... 得分: {keyword_score}, 匹配: {matched_keywords}")

            # 只保留匹配度 > 0.25 的论文
            if keyword_score > 0.25:
                scored_papers.append({
                    "paper_id": paper_id,
                    "title": paper_node["title"],
                    "keyword_score": keyword_score,
                    "matched_keywords": matched_keywords,
                    "match_rate": f"{len(matched_keywords)}/{len(query_keywords)}"
                })
            else:
                logger.debug(f"论文 {paper_id[:30]}... 得分过低 ({keyword_score}), 被过滤")

        # 按得分排序
        scored_papers.sort(key=lambda x: x["keyword_score"], reverse=True)

        logger.info(f"评分完成，{len(scored_papers)} 篇论文通过阈值 (>0.25)")

        return scored_papers

    def _calculate_keyword_match_score(self, query_keywords: List[str], paper_keywords: dict) -> tuple:
        """计算关键词匹配得分 - 要求尽可能完全匹配，特定TMS类型严格匹配"""

        if not query_keywords:
            return 0.0, []

        # 将分类关键词展平
        all_paper_keywords = []
        for kw_list in paper_keywords.values():
            if kw_list:
                all_paper_keywords.extend(kw_list)

        if not all_paper_keywords:
            return 0.0, []

        logger.debug(f"查询关键词: {query_keywords}")
        logger.debug(f"论文关键词: {all_paper_keywords[:20]}")  # 只显示前20个

        matched_keywords = []
        match_scores = []

        for q_keyword in query_keywords:
            best_match_score = 0.0
            best_match_keyword = None

            # 检查是否为特定TMS类型
            is_specific_tms = self._is_specific_tms_type(q_keyword)

            # 获取映射关键词
            mapped_keywords = self._get_mapped_keywords(q_keyword)
            all_query_variants = [q_keyword] + mapped_keywords

            logger.debug(f"处理查询关键词 '{q_keyword}', 变体: {all_query_variants[:10]}")

            for p_keyword in all_paper_keywords:
                p_keyword_lower = p_keyword.lower()

                # 检查所有变体
                for q_variant in all_query_variants:
                    q_variant_lower = q_variant.lower()

                    # 如果查询是特定TMS类型，要求严格匹配
                    if is_specific_tms:
                        if not self._tms_type_matches(q_variant_lower, p_keyword_lower):
                            continue

                    # 1. 精确匹配 (1.0分)
                    if q_variant_lower == p_keyword_lower:
                        match_score = 1.0
                        logger.debug(f"  精确匹配: '{q_variant}' == '{p_keyword}'")
                    # 2. 完全包含 (0.9分)
                    elif q_variant_lower in p_keyword_lower or p_keyword_lower in q_variant_lower:
                        match_score = 0.9
                        logger.debug(f"  包含匹配: '{q_variant}' <-> '{p_keyword}'")
                    # 3. 高相似度 (0.8分)
                    elif self._calculate_keyword_similarity_simple(q_variant_lower, p_keyword_lower) > 0.85:
                        match_score = 0.8
                        logger.debug(f"  相似匹配: '{q_variant}' ~ '{p_keyword}'")
                    else:
                        match_score = 0.0

                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_keyword = p_keyword

            if best_match_score > 0:
                matched_keywords.append(best_match_keyword)
                match_scores.append(best_match_score)
                logger.debug(f"关键词 '{q_keyword}' 最佳匹配: '{best_match_keyword}' (得分: {best_match_score})")
            else:
                logger.debug(f"关键词 '{q_keyword}' 无匹配")

        # 计算匹配率：匹配的关键词数 / 查询关键词总数
        if not match_scores:
            return 0.0, []

        # 综合得分 = 平均匹配质量 * 匹配覆盖率
        avg_quality = sum(match_scores) / len(match_scores)
        coverage = len(matched_keywords) / len(query_keywords)

        final_score = round(avg_quality * coverage, 3)

        logger.info(
            f"最终匹配: {len(matched_keywords)}/{len(query_keywords)}, 质量: {avg_quality:.3f}, 覆盖率: {coverage:.3f}, 得分: {final_score}")

        return final_score, matched_keywords

    def _is_specific_tms_type(self, keyword: str) -> bool:
        """判断是否为特定TMS类型（使用已有映射表）"""
        keyword_lower = keyword.lower()

        # 使用INTERVENTION_SYNONYMS中的特定TMS类型
        specific_tms_types = [
            "rtms", "dtms", "atms", "itbs", "ctbs", "tbs",
            "quadripulse", "pas", "aitbs", "actbs", "atbs"
        ]

        # 检查是否匹配特定类型或其同义词
        for tms_type in specific_tms_types:
            if tms_type in INTERVENTION_SYNONYMS:
                synonyms = [s.lower() for s in INTERVENTION_SYNONYMS[tms_type]]
                if any(syn in keyword_lower for syn in synonyms):
                    return True

        return False

    def _tms_type_matches(self, query_tms: str, paper_tms: str) -> bool:
        """检查TMS类型是否匹配（使用已有映射表）"""

        # 找到查询TMS所属的类型
        query_type = None
        for tms_type, synonyms in INTERVENTION_SYNONYMS.items():
            synonyms_lower = [s.lower() for s in synonyms]
            if any(syn in query_tms for syn in synonyms_lower):
                query_type = tms_type
                break

        if not query_type:
            return True  # 如果不是特定类型，允许匹配

        # 检查论文TMS是否属于同一类型（使用映射表）
        if query_type in INTERVENTION_SYNONYMS:
            paper_synonyms = [s.lower() for s in INTERVENTION_SYNONYMS[query_type]]
            return any(syn in paper_tms for syn in paper_synonyms)

        return True

    async def _get_keyword_match_details(self, query_keywords: List[str]) -> dict:
        """获取关键词匹配详情"""

        # 获取关键词索引
        keyword_keys = await self.keyword_index.all_keys()
        keyword_index = {}

        if keyword_keys:
            keyword_values = await self.keyword_index.get_by_ids(keyword_keys)
            keyword_index = {key: value for key, value in zip(keyword_keys, keyword_values) if value is not None}

        match_details = {}

        for keyword in query_keywords:
            if keyword in keyword_index:
                match_details[keyword] = {
                    "exact_matches": len(keyword_index[keyword]["papers"]),
                    "papers": keyword_index[keyword]["papers"][:5]
                }
            else:
                # 查找相似关键词
                similar_keywords = []
                for stored_keyword in keyword_index.keys():
                    if self._calculate_keyword_similarity_simple(keyword, stored_keyword) > 0.7:
                        similar_keywords.append(stored_keyword)

                match_details[keyword] = {
                    "exact_matches": 0,
                    "similar_keywords": similar_keywords[:3]
                }

        return match_details

    async def _structural_query(self, query: str, candidate_paper_ids: set = None) -> dict:
        """结构化查询 - 基于论文的结构化特征进行匹配，使用BM25思想调整权重"""

        # 1. 解析查询中的结构化信息
        structural_criteria = await self._parse_structural_query(query)

        if not structural_criteria:
            return {"query": query, "mode": "structural", "papers": []}

        # 2. 获取所有论文节点
        paper_keys = await self.paper_nodes_storage.all_keys()

        # 如果指定了候选范围，只处理候选论文
        if candidate_paper_ids is not None:
            paper_keys = [key for key in paper_keys if key in candidate_paper_ids]
            logger.info(f"结构化查询限制在 {len(paper_keys)} 篇候选论文范围内")

        all_papers = {}

        if paper_keys:
            paper_values = await self.paper_nodes_storage.get_by_ids(paper_keys)
            all_papers = {key: value for key, value in zip(paper_keys, paper_values) if value is not None}

        # 计算特征的IDF权重
        feature_idf_weights = await self._calculate_feature_idf_weights(all_papers, structural_criteria)

        # 3. 计算结构化匹配得分
        scored_papers = []

        has_condition_criteria = "condition" in structural_criteria

        for paper_id, paper_node in all_papers.items():
            if has_condition_criteria:
                if not self._check_condition_match(structural_criteria, paper_node["structured_features"]):
                    logger.debug(f"论文 {paper_id[:30]}... 疾病不匹配，跳过")
                    continue

            match_score = self._calculate_structural_match_score_with_idf(
                structural_criteria, 
                paper_node["structured_features"],
                feature_idf_weights  # 传入IDF权重
            )

            if match_score > 0.6:
                scored_papers.append({
                    "paper_id": paper_id,
                    "title": paper_node["title"],
                    "structural_score": match_score,
                    "matched_criteria": self._get_matched_criteria(
                        structural_criteria, paper_node["structured_features"]
                    )
                })

        # 4. 排序
        scored_papers.sort(key=lambda x: x["structural_score"], reverse=True)

        return {
            "query": query,
            "mode": "structural",
            "structural_criteria": structural_criteria,
            "total_papers": len(scored_papers),
            "papers": scored_papers
        }

    def _check_condition_match(self, criteria: dict, features: dict) -> bool:
        """检查疾病是否匹配（用于预过滤）"""

        if "condition" not in criteria:
            return True  # 没有疾病条件，不过滤

        participants = features.get("participants", {})
        if not participants:
            return False

        feature_condition = participants.get("condition")
        if not feature_condition:
            return False

        feature_condition = str(feature_condition).lower()

        query_conditions = criteria["condition"]
        if not isinstance(query_conditions, list):
            query_conditions = [query_conditions]

        # 检查是否有任何一个查询疾病匹配
        for qc in query_conditions:
            if self._match_condition(qc, feature_condition):
                return True

        return False

    async def _calculate_feature_idf_weights(self, all_papers: dict, criteria: dict) -> dict:
        """
        计算结构化特征的IDF权重

        IDF(特征) = log(总论文数 / 包含该特征的论文数)
        """
        import math

        total_papers = len(all_papers)

        if total_papers == 0:
            return {}

        # 统计每个特征值出现的论文数
        feature_counts = {
            "intervention_type": {},
            "frequency": {},
            "target_region": {},
            "condition": {},
            "study_type": {}
        }

        for paper_id, paper_node in all_papers.items():
            features = paper_node.get("structured_features", {})

            # 统计干预类型
            intervention = features.get("intervention", {})
            intervention_type = intervention.get("type") if intervention else None
            if intervention_type:
                intervention_type = str(intervention_type).lower()
                feature_counts["intervention_type"][intervention_type] = \
                    feature_counts["intervention_type"].get(intervention_type, 0) + 1

            # 统计频率（处理字符串格式）
            frequency = intervention.get("frequency") if intervention else None
            if frequency:
                # 将频率归类到区间（避免过于细分）
                freq_bucket = self._get_frequency_bucket(frequency)
                if freq_bucket != "unknown_freq":  # 只统计能解析的频率
                    feature_counts["frequency"][freq_bucket] = \
                        feature_counts["frequency"].get(freq_bucket, 0) + 1

            # 统计靶点（可能需要标准化）
            target = intervention.get("target") if intervention else None
            if target:
                target = str(target).lower()
                # 标准化靶点名称（去除"left"/"right"等修饰词）
                normalized_target = self._normalize_target_name(target)
                feature_counts["target_region"][normalized_target] = \
                    feature_counts["target_region"].get(normalized_target, 0) + 1

            # 统计疾病（可能需要标准化）
            participants = features.get("participants", {})
            condition = participants.get("condition") if participants else None
            if condition:
                condition = str(condition).lower()
                # 标准化疾病名称
                normalized_condition = self._normalize_condition_name(condition)
                feature_counts["condition"][normalized_condition] = \
                    feature_counts["condition"].get(normalized_condition, 0) + 1

            # 统计研究设计
            design = features.get("design", {})
            study_type = design.get("study_type") if design else None
            if study_type:
                study_type = str(study_type).lower()
                feature_counts["study_type"][study_type] = \
                    feature_counts["study_type"].get(study_type, 0) + 1

        # 计算IDF权重
        idf_weights = {}

        for feature_type, counts in feature_counts.items():
            idf_weights[feature_type] = {}
            for feature_value, count in counts.items():
                # IDF = log(N / df)，加0.5避免除零
                idf = math.log((total_papers + 0.5) / (count + 0.5))
                idf_weights[feature_type][feature_value] = idf

        logger.info(f"计算IDF权重完成，示例: {self._get_idf_sample(idf_weights)}")

        return idf_weights

    def _normalize_target_name(self, target: str) -> str:
        """标准化靶点名称 - 使用配置文件中的映射"""
        target = target.lower().strip()

        # 去除左右半球修饰词
        target = re.sub(r'\b(left|right|bilateral)\s+', '', target)

        # 使用BRAIN_REGIONS映射表
        for standard_name, aliases in BRAIN_REGIONS.items():
            aliases_lower = [a.lower() for a in aliases]
            if target in aliases_lower or any(alias in target for alias in aliases_lower):
                return standard_name

        return target

    def _normalize_condition_name(self, condition: str) -> str:
        """标准化疾病名称 - 使用配置文件中的映射"""
        condition = condition.lower().strip()

        # 使用DISEASE_CONDITIONS映射表
        for standard_name, aliases in DISEASE_CONDITIONS.items():
            aliases_lower = [a.lower() for a in aliases]
            if condition in aliases_lower or any(alias in condition for alias in aliases_lower):
                return standard_name

        return condition

    def _get_frequency_bucket(self, frequency) -> str:
        """
        将频率归类到区间

        处理多种频率格式：
        - "20 Hz" -> 提取20
        - "3*50Hz pulses at 5Hz bursts" -> 提取主频率5Hz
        - 10 (数字) -> 直接使用
        """
        if frequency is None:
            return "unknown_freq"

        # 如果已经是数字，直接使用
        if isinstance(frequency, (int, float)):
            freq_value = float(frequency)
        else:
            # 字符串格式，需要解析
            freq_str = str(frequency).lower()

            # 1. TBS特殊格式: "50Hz bursts at 5Hz" 或 "3*50Hz pulses at 5Hz bursts"
            # 主频率是delivery frequency (5Hz)，而不是burst frequency (50Hz)
            tbs_pattern = r'(?:at|delivered\s+at)\s+(\d+(?:\.\d+)?)\s*hz'
            tbs_match = re.search(tbs_pattern, freq_str)
            if tbs_match:
                freq_value = float(tbs_match.group(1))
            else:
                # 2. 普通格式: "20 Hz", "10Hz", "1.5 hz"
                normal_pattern = r'(\d+(?:\.\d+)?)\s*hz'
                normal_match = re.search(normal_pattern, freq_str)
                if normal_match:
                    freq_value = float(normal_match.group(1))
                else:
                    # 3. 无法解析，返回未知
                    logger.warning(f"无法解析频率格式: {frequency}")
                    return "unknown_freq"

        # 根据频率值分桶
        if freq_value <= 1:
            return "low_freq_<=1hz"
        elif freq_value <= 5:
            return "mid_freq_1-5hz"
        elif freq_value <= 10:
            return "high_freq_5-10hz"
        elif freq_value <= 20:
            return "high_freq_10-20hz"
        else:
            return "very_high_freq_>20hz"

    def _get_idf_sample(self, idf_weights: dict) -> dict:
        """获取IDF权重示例（用于日志）"""
        sample = {}
        for feature_type, weights in idf_weights.items():
            if weights:
                # 取权重最高和最低的各一个
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                sample[feature_type] = {
                    "highest": sorted_weights[0] if sorted_weights else None,
                    "lowest": sorted_weights[-1] if sorted_weights else None
                }
        return sample

    def _calculate_structural_match_score_with_idf(
            self,
            criteria: dict,
            features: dict,
            idf_weights: dict
    ) -> float:
        """
        计算结构化匹配得分 - 使用IDF权重调整

        核心改进：
        1. 常见特征（如"TMS"、"抑郁症"）权重低
        2. 罕见特征（如"iTBS"、"10Hz"）权重高
        3. 动态调整各维度的贡献
        """

        if not criteria or not features:
            return 0.0

        total_score = 0.0
        total_weight = 0.0  # 动态权重总和

        # 1. 干预类型匹配
        if "intervention_type" in criteria:
            intervention = features.get("intervention", {})
            if intervention:
                feature_type = intervention.get("type")
                if feature_type:
                    feature_type = str(feature_type).lower()

                    query_types = criteria["intervention_type"]
                    if not isinstance(query_types, list):
                        query_types = [query_types]

                    # 检查是否匹配
                    matched = False
                    matched_type = None
                    for qt in query_types:
                        if self._match_intervention_type(qt, feature_type):
                            matched = True
                            matched_type = feature_type
                            break

                    if matched and matched_type:
                        # 使用IDF权重
                        idf_weight = idf_weights.get("intervention_type", {}).get(matched_type, 1.0)
                        # 归一化IDF权重到0.5-2.0范围
                        normalized_idf = min(max(idf_weight, 0.5), 2.0)

                        dynamic_weight = 0.3 * normalized_idf
                        total_weight += dynamic_weight
                        total_score += dynamic_weight

                        logger.debug(f"干预类型匹配: {matched_type}, IDF={idf_weight:.2f}, 权重={dynamic_weight:.3f}")

        # 2. 频率匹配
        if any(k.startswith("frequency") for k in criteria.keys()):
            intervention = features.get("intervention", {})
            if intervention:
                feature_freq = intervention.get("frequency")

                if feature_freq:
                    freq_bucket = self._get_frequency_bucket(feature_freq)
                    idf_weight = idf_weights.get("frequency", {}).get(freq_bucket, 1.0)
                    normalized_idf = min(max(idf_weight, 0.5), 2.0)

                    if self._match_frequency_advanced(criteria, intervention):
                        dynamic_weight = 0.25 * normalized_idf
                        total_weight += dynamic_weight
                        total_score += dynamic_weight

                        logger.debug(
                            f"频率匹配: {feature_freq} ({freq_bucket}), IDF={idf_weight:.2f}, 权重={dynamic_weight:.3f}")

        # 3. 脑区匹配
        if "target_region" in criteria:
            intervention = features.get("intervention", {})
            if intervention:
                feature_target = intervention.get("target")
                if feature_target:
                    feature_target = str(feature_target).lower()

                    query_regions = criteria["target_region"]
                    if not isinstance(query_regions, list):
                        query_regions = [query_regions]

                    matched = False
                    for qr in query_regions:
                        if self._match_brain_region(qr, feature_target):
                            matched = True
                            break

                    if matched:
                        # 使用标准化后的靶点名称获取IDF
                        normalized_target = self._normalize_target_name(feature_target)
                        idf_weight = idf_weights.get("target_region", {}).get(normalized_target, 1.0)
                        normalized_idf = min(max(idf_weight, 0.5), 2.0)

                        dynamic_weight = 0.25 * normalized_idf
                        total_weight += dynamic_weight
                        total_score += dynamic_weight

                        logger.debug(
                            f"脑区匹配: {feature_target} -> {normalized_target}, IDF={idf_weight:.2f}, 权重={dynamic_weight:.3f}")

        # 4. 疾病匹配
        if "condition" in criteria:
            participants = features.get("participants", {})
            if participants:
                feature_condition = participants.get("condition")
                if feature_condition:
                    feature_condition = str(feature_condition).lower()

                    query_conditions = criteria["condition"]
                    if not isinstance(query_conditions, list):
                        query_conditions = [query_conditions]

                    matched = False
                    for qc in query_conditions:
                        if self._match_condition(qc, feature_condition):
                            matched = True
                            break

                    if matched:
                        # 使用标准化后的疾病名称获取IDF
                        normalized_condition = self._normalize_condition_name(feature_condition)
                        idf_weight = idf_weights.get("condition", {}).get(normalized_condition, 1.0)
                        normalized_idf = min(max(idf_weight, 0.5), 2.0)

                        dynamic_weight = 0.15 * normalized_idf
                        total_weight += dynamic_weight
                        total_score += dynamic_weight

                        logger.debug(
                            f"疾病匹配: {feature_condition} -> {normalized_condition}, IDF={idf_weight:.2f}, 权重={dynamic_weight:.3f}")

        # 5. 研究设计匹配
        if "study_type" in criteria:
            design = features.get("design", {})
            if design:
                feature_study_type = design.get("study_type")
                if feature_study_type:
                    feature_study_type = str(feature_study_type).lower()

                    query_study_types = criteria["study_type"]
                    if not isinstance(query_study_types, list):
                        query_study_types = [query_study_types]

                    matched = False
                    for qst in query_study_types:
                        if self._match_study_type(qst, feature_study_type):
                            matched = True
                            break

                    if matched:
                        idf_weight = idf_weights.get("study_type", {}).get(feature_study_type, 1.0)
                        normalized_idf = min(max(idf_weight, 0.5), 2.0)

                        dynamic_weight = 0.05 * normalized_idf
                        total_weight += dynamic_weight
                        total_score += dynamic_weight

        # 归一化得分
        if total_weight > 0:
            final_score = min(total_score, 1.0)
            logger.debug(f"结构化匹配最终得分: {final_score:.3f} (总权重: {total_weight:.3f})")
            return final_score
        else:
            return 0.0

    async def _parse_structural_query(self, query: str) -> dict:
        """解析查询中的结构化信息 - 支持多条件提取"""

        criteria = {}
        query_lower = query.lower()

        # 解析干预类型（可能有多个）
        intervention_types = []
        for intervention, keywords in INTERVENTION_HIERARCHY.items():
            if any(kw in query_lower for kw in keywords):
                intervention_types.append(intervention)

        if intervention_types:
            # 如果只有一个，直接使用；如果有多个，保存为列表
            criteria["intervention_type"] = intervention_types[0] if len(intervention_types) == 1 else intervention_types

        # 解析频率（可能有多个频率条件）
        frequency_info = self._parse_frequency_from_query(query_lower)
        if frequency_info:
            criteria.update(frequency_info)

        # 解析脑区（可能有多个）
        target_regions = []
        for region, keywords in BRAIN_REGIONS.items():
            if any(kw in query_lower for kw in keywords):
                target_regions.append(region)

        if target_regions:
            criteria["target_region"] = target_regions[0] if len(target_regions) == 1 else target_regions

        # 解析疾病/人群（可能有多个）
        conditions = []
        for condition, keywords in DISEASE_CONDITIONS.items():
            if any(kw in query_lower for kw in keywords):
                conditions.append(condition)

        if conditions:
            criteria["condition"] = conditions[0] if len(conditions) == 1 else conditions

        # 解析研究设计（可能有多个）
        study_types = []
        for study_type, keywords in STUDY_DESIGNS.items():
            if any(kw in query_lower for kw in keywords):
                study_types.append(study_type)

        if study_types:
            criteria["study_type"] = study_types[0] if len(study_types) == 1 else study_types

        # 解析样本量
        sample_pattern = r"(\d+)\s*(?:participants|subjects|patients)"
        match = re.search(sample_pattern, query_lower)
        if match:
            criteria["min_sample_size"] = int(match.group(1))

        return criteria

    def _parse_frequency_from_query(self, query_lower: str) -> dict:
        """从查询中解析频率信息"""

        freq_info = {}

        # 1. TBS特殊模式: "50Hz bursts at 5Hz"
        tbs_pattern = r'(\d+)\s*hz\s+bursts?\s+(?:at|delivered\s+at)\s+(\d+)\s*hz'
        tbs_match = re.search(tbs_pattern, query_lower)
        if tbs_match:
            burst_freq = int(tbs_match.group(1))
            delivery_freq = int(tbs_match.group(2))
            freq_info["frequency_type"] = "tbs"
            freq_info["burst_frequency"] = burst_freq
            freq_info["delivery_frequency"] = delivery_freq
            return freq_info

        # 2. 频率范围: "5-10Hz", "10-20Hz"
        range_pattern = r'(\d+)\s*-\s*(\d+)\s*hz'
        range_match = re.search(range_pattern, query_lower)
        if range_match:
            min_freq = int(range_match.group(1))
            max_freq = int(range_match.group(2))
            freq_info["frequency_type"] = "range"
            freq_info["frequency_min"] = min_freq
            freq_info["frequency_max"] = max_freq
            return freq_info

        # 3. 精确频率: "10Hz", "1Hz"
        exact_pattern = r'(\d+(?:\.\d+)?)\s*hz'
        exact_match = re.search(exact_pattern, query_lower)
        if exact_match:
            freq_value = float(exact_match.group(1))
            freq_info["frequency_type"] = "exact"
            freq_info["frequency"] = freq_value
            return freq_info

        # 4. 描述性频率: "high frequency", "low frequency"
        if re.search(r'\bhigh[- ]frequency\b', query_lower):
            freq_info["frequency_type"] = "high"
            freq_info["frequency_min"] = 5  # 高频通常 ≥5Hz
            return freq_info

        if re.search(r'\blow[- ]frequency\b', query_lower):
            freq_info["frequency_type"] = "low"
            freq_info["frequency_max"] = 1  # 低频通常 ≤1Hz
            return freq_info

        return freq_info

    def _calculate_structural_match_score(self, criteria: dict, features: dict) -> float:
        """计算结构化匹配得分 - 支持多条件匹配"""

        if not criteria or not features:
            return 0.0

        total_score = 0.0
        max_possible_score = 0.0

        # 干预类型匹配 (权重: 0.3)
        if "intervention_type" in criteria:
            max_possible_score += 0.3
            intervention = features.get("intervention", {})
            feature_type = intervention.get("type", "")

            query_types = criteria["intervention_type"]
            # 支持单个或多个干预类型
            if isinstance(query_types, list):
                # 多个条件：只要匹配任意一个即可
                if any(self._match_intervention_type(qt, feature_type) for qt in query_types):
                    total_score += 0.3
            else:
                # 单个条件
                if self._match_intervention_type(query_types, feature_type):
                    total_score += 0.3

        # 频率匹配 (权重: 0.2)
        if any(k.startswith("frequency") for k in criteria.keys()):
            max_possible_score += 0.2
            intervention = features.get("intervention", {})
            if self._match_frequency_advanced(criteria, intervention):
                total_score += 0.2

        # 脑区匹配 (权重: 0.25)
        if "target_region" in criteria:
            max_possible_score += 0.25
            intervention = features.get("intervention", {})
            feature_target = intervention.get("target", "")

            query_regions = criteria["target_region"]
            # 支持单个或多个脑区
            if isinstance(query_regions, list):
                # 多个条件：只要匹配任意一个即可
                if any(self._match_brain_region(qr, feature_target) for qr in query_regions):
                    total_score += 0.25
            else:
                # 单个条件
                if self._match_brain_region(query_regions, feature_target):
                    total_score += 0.25

        # 疾病/人群匹配 (权重: 0.15)
        if "condition" in criteria:
            max_possible_score += 0.15
            participants = features.get("participants", {})
            feature_condition = participants.get("condition", "")

            query_conditions = criteria["condition"]
            # 支持单个或多个疾病
            if isinstance(query_conditions, list):
                # 多个条件：只要匹配任意一个即可
                if any(self._match_condition(qc, feature_condition) for qc in query_conditions):
                    total_score += 0.15
            else:
                # 单个条件
                if self._match_condition(query_conditions, feature_condition):
                    total_score += 0.15

        # 研究设计匹配 (权重: 0.1)
        if "study_type" in criteria:
            max_possible_score += 0.1
            design = features.get("design", {})
            feature_study_type = design.get("study_type", "")

            query_study_types = criteria["study_type"]
            # 支持单个或多个研究设计
            if isinstance(query_study_types, list):
                if any(self._match_study_type(qst, feature_study_type) for qst in query_study_types):
                    total_score += 0.1
            else:
                if self._match_study_type(query_study_types, feature_study_type):
                    total_score += 0.1

        # 样本量匹配
        if "min_sample_size" in criteria:
            max_possible_score += 0.05
            participants = features.get("participants", {})
            sample_size = participants.get("sample_size", 0)
            if sample_size >= criteria["min_sample_size"]:
                total_score += 0.05

        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    def _match_intervention_type(self, criteria_type: str, feature_type: str) -> bool:
        """匹配干预类型"""
        if not feature_type:
            return False

        feature_lower = feature_type.lower()
        criteria_lower = criteria_type.lower()

        for intervention, keywords in INTERVENTION_HIERARCHY.items():
            if criteria_lower == intervention:
                return any(kw in feature_lower for kw in keywords)

        return criteria_lower in feature_lower

    def _match_frequency_advanced(self, criteria: dict, intervention: dict) -> bool:
        """高级频率匹配 - 支持多种频率模式"""

        if not intervention:
            return False

        feature_freq = intervention.get("frequency")
        if not feature_freq:
            return False

        # 提取论文中的频率信息
        paper_freq_info = self._extract_frequency_from_feature(feature_freq)
        if not paper_freq_info:
            return False

        freq_type = criteria.get("frequency_type")

        # 1. TBS模式匹配
        if freq_type == "tbs":
            return self._match_tbs_frequency(criteria, paper_freq_info)

        # 2. 范围匹配
        elif freq_type == "range":
            return self._match_frequency_range(
                criteria["frequency_min"],
                criteria["frequency_max"],
                paper_freq_info
            )

        # 3. 精确匹配（允许±2Hz误差）
        elif freq_type == "exact":
            query_freq = criteria["frequency"]
            paper_freq = paper_freq_info.get("main_frequency")
            if paper_freq:
                return abs(paper_freq - query_freq) <= 2

        # 4. 高频匹配
        elif freq_type == "high":
            paper_freq = paper_freq_info.get("main_frequency")
            if paper_freq:
                return paper_freq >= 5

        # 5. 低频匹配
        elif freq_type == "low":
            paper_freq = paper_freq_info.get("main_frequency")
            if paper_freq:
                return paper_freq <= 1

        return False

    def _extract_frequency_from_feature(self, feature_freq) -> dict:
        """从论文特征中提取频率信息"""

        if isinstance(feature_freq, (int, float)):
            return {"main_frequency": float(feature_freq)}

        if not isinstance(feature_freq, str):
            return {}

        freq_lower = feature_freq.lower()
        freq_info = {}

        # 1. TBS模式: "50Hz bursts at 5Hz"
        tbs_pattern = r'(\d+)\s*hz\s+bursts?\s+(?:at|delivered\s+at)\s+(\d+)\s*hz'
        tbs_match = re.search(tbs_pattern, freq_lower)
        if tbs_match:
            freq_info["type"] = "tbs"
            freq_info["burst_frequency"] = int(tbs_match.group(1))
            freq_info["delivery_frequency"] = int(tbs_match.group(2))
            freq_info["main_frequency"] = int(tbs_match.group(2))  # 主频率用delivery频率
            return freq_info

        # 2. 范围: "5-10Hz"
        range_pattern = r'(\d+)\s*-\s*(\d+)\s*hz'
        range_match = re.search(range_pattern, freq_lower)
        if range_match:
            min_freq = int(range_match.group(1))
            max_freq = int(range_match.group(2))
            freq_info["type"] = "range"
            freq_info["min"] = min_freq
            freq_info["max"] = max_freq
            freq_info["main_frequency"] = (min_freq + max_freq) / 2
            return freq_info

        # 3. 精确频率: "10Hz"
        exact_pattern = r'(\d+(?:\.\d+)?)\s*hz'
        exact_match = re.search(exact_pattern, freq_lower)
        if exact_match:
            freq_info["type"] = "exact"
            freq_info["main_frequency"] = float(exact_match.group(1))
            return freq_info

        # 4. 描述性
        if "high" in freq_lower and "frequency" in freq_lower:
            freq_info["type"] = "high"
            freq_info["main_frequency"] = 10  # 假设高频为10Hz
            return freq_info

        if "low" in freq_lower and "frequency" in freq_lower:
            freq_info["type"] = "low"
            freq_info["main_frequency"] = 1  # 假设低频为1Hz
            return freq_info

        return freq_info

    def _match_tbs_frequency(self, criteria: dict, paper_freq_info: dict) -> bool:
        """匹配TBS频率"""

        if paper_freq_info.get("type") != "tbs":
            return False

        # 检查burst频率和delivery频率是否匹配
        query_burst = criteria.get("burst_frequency")
        query_delivery = criteria.get("delivery_frequency")

        paper_burst = paper_freq_info.get("burst_frequency")
        paper_delivery = paper_freq_info.get("delivery_frequency")

        burst_match = abs(paper_burst - query_burst) <= 5 if query_burst and paper_burst else True
        delivery_match = abs(paper_delivery - query_delivery) <= 1 if query_delivery and paper_delivery else True

        return burst_match and delivery_match

    def _match_frequency_range(self, min_freq: float, max_freq: float, paper_freq_info: dict) -> bool:
        """匹配频率范围"""

        paper_freq = paper_freq_info.get("main_frequency")
        if not paper_freq:
            return False

        # 论文频率在查询范围内
        if min_freq <= paper_freq <= max_freq:
            return True

        # 如果论文也是范围，检查是否有重叠
        if paper_freq_info.get("type") == "range":
            paper_min = paper_freq_info.get("min")
            paper_max = paper_freq_info.get("max")
            # 检查范围重叠
            return not (paper_max < min_freq or paper_min > max_freq)

        return False

    def _match_brain_region(self, criteria_region: str, feature_target: str) -> bool:
        """匹配脑区"""
        if not feature_target:
            return False

        feature_lower = feature_target.lower()
        criteria_lower = criteria_region.lower()

        for region, keywords in BRAIN_REGIONS.items():
            if criteria_lower == region:
                return any(kw in feature_lower for kw in keywords)

        return criteria_lower in feature_lower

    def _match_condition(self, criteria_condition: str, feature_condition: str) -> bool:
        """匹配疾病条件"""
        if not feature_condition:
            return False

        feature_lower = feature_condition.lower()
        criteria_lower = criteria_condition.lower()

        for condition, keywords in DISEASE_CONDITIONS.items():
            if criteria_lower == condition:
                return any(kw in feature_lower for kw in keywords)

        return criteria_lower in feature_lower

    def _match_study_type(self, query_type: str, feature_type: str) -> bool:
        """匹配研究设计类型"""
        if not feature_type:
            return False

        query_lower = query_type.lower()
        feature_lower = feature_type.lower()

        return query_lower in feature_lower or feature_lower in query_lower

    def _match_blinding(self, criteria_blinding: str, feature_blinding: str) -> bool:
        """匹配盲法"""
        if not feature_blinding:
            return False

        return criteria_blinding.lower() in feature_blinding.lower()

    def _get_matched_criteria(self, criteria: dict, features: dict) -> dict:
        """获取匹配的标准详情 - 支持多条件"""

        matched = {}

        for key, value in criteria.items():
            if key == "intervention_type":
                intervention = features.get("intervention", {})
                feature_type = intervention.get("type", "")

                # 处理单个或多个干预类型
                if isinstance(value, list):
                    matched_types = [v for v in value if self._match_intervention_type(v, feature_type)]
                    if matched_types:
                        matched[key] = {
                            "criteria": matched_types,
                            "matched": feature_type
                        }
                else:
                    if self._match_intervention_type(value, feature_type):
                        matched[key] = {"criteria": value, "matched": feature_type}

            elif key.startswith("frequency"):
                intervention = features.get("intervention", {})
                if self._match_frequency_advanced(criteria, intervention):
                    matched["frequency"] = {
                        "criteria": self._format_frequency_criteria(criteria),
                        "matched": intervention.get("frequency", "")
                    }

            elif key == "target_region":
                intervention = features.get("intervention", {})
                feature_target = intervention.get("target", "")

                # 处理单个或多个脑区
                if isinstance(value, list):
                    matched_regions = [v for v in value if self._match_brain_region(v, feature_target)]
                    if matched_regions:
                        matched[key] = {
                            "criteria": matched_regions,
                            "matched": feature_target
                        }
                else:
                    if self._match_brain_region(value, feature_target):
                        matched[key] = {"criteria": value, "matched": feature_target}

            elif key == "condition":
                participants = features.get("participants", {})
                feature_condition = participants.get("condition", "")

                # 处理单个或多个疾病
                if isinstance(value, list):
                    matched_conditions = [v for v in value if self._match_condition(v, feature_condition)]
                    if matched_conditions:
                        matched[key] = {
                            "criteria": matched_conditions,
                            "matched": feature_condition
                        }
                else:
                    if self._match_condition(value, feature_condition):
                        matched[key] = {"criteria": value, "matched": feature_condition}

            elif key == "study_type":
                design = features.get("design", {})
                feature_study_type = design.get("study_type", "")

                # 处理单个或多个研究设计
                if isinstance(value, list):
                    matched_types = [v for v in value if self._match_study_type(v, feature_study_type)]
                    if matched_types:
                        matched[key] = {
                            "criteria": matched_types,
                            "matched": feature_study_type
                        }
                else:
                    if self._match_study_type(value, feature_study_type):
                        matched[key] = {"criteria": value, "matched": feature_study_type}

        return matched

    def _format_frequency_criteria(self, criteria: dict) -> str:
        """格式化频率条件为可读字符串"""

        freq_type = criteria.get("frequency_type")

        if freq_type == "tbs":
            burst = criteria.get("burst_frequency")
            delivery = criteria.get("delivery_frequency")
            return f"{burst}Hz bursts at {delivery}Hz"

        elif freq_type == "range":
            min_f = criteria.get("frequency_min")
            max_f = criteria.get("frequency_max")
            return f"{min_f}-{max_f}Hz"

        elif freq_type == "exact":
            freq = criteria.get("frequency")
            return f"{freq}Hz"

        elif freq_type == "high":
            return "high frequency (≥5Hz)"

        elif freq_type == "low":
            return "low frequency (≤1Hz)"

        return "unknown frequency"

    async def query_and_meta_analyze(self, query: str, outcome_name: str = None,
                                     save_plot: bool = True, plot_path: str = None,
                                     query_mode: str = "comprehensive", **meta_kwargs) -> dict:
        """查询论文并进行Meta分析
        args_mode:
        community_based
        structural
        semantic
        keyword
        comprehensive
        """

        logger.info(f"开始查询和Meta分析: {query}")

        # 第1步：查询相关论文
        query_results = await self.query_papers(query, mode=query_mode)

        if not query_results.get("papers"):
            return {"error": "未找到相关论文", "query_results": query_results}

        # 第2步：提取论文ID
        paper_ids = [paper["paper_id"] for paper in query_results["papers"]]
        logger.info(f"查询到 {len(paper_ids)} 篇相关论文")

        # 第3步：创建MetaAnalyzer并进行分析
        analyzer = MetaAnalyzer(self.meta_graphrag)
        await analyzer.load_papers_data()  # 异步加载数据

        meta_results = {}
        failed_outcomes = {}
        forest_plots = {}

        try:
            # 如果没有指定结局指标，先查看可用的
            if not outcome_name:
                outcomes, viable_outcomes = analyzer.get_available_outcomes_for_papers(paper_ids)
                if not viable_outcomes:
                    return {
                        "error": "指定论文中没有足够数据进行Meta分析的结局指标",
                        "query_results": query_results,
                        "available_outcomes": outcomes
                    }

                # 对每个可行的结局指标分别进行Meta分析
                for outcome in viable_outcomes:
                    try:
                        logger.info(f"开始分析结局指标: {outcome}")
                        meta_result = await analyzer.perform_targeted_meta_analysis(
                            target_paper_ids=paper_ids,
                            outcome_name=outcome,
                            query = query,
                            **meta_kwargs
                        )
                        meta_results[outcome] = meta_result
                        logger.info(f"结局指标 {outcome} 分析成功")

                        # 立即生成森林图
                        if save_plot:
                            safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")[:30]
                            safe_outcome = outcome.replace(" ", "_").replace("/", "_")[:20]
                            outcome_plot_path = f"forest_plot_{safe_query}_{safe_outcome}.png"

                            try:
                                analyzer.generate_forest_plot(meta_result, outcome_plot_path)
                                forest_plots[outcome] = outcome_plot_path
                                logger.info(f"森林图已保存: {outcome_plot_path}")
                            except Exception as plot_e:
                                logger.error(f"生成森林图失败 ({outcome}): {plot_e}")

                        # 打印Meta分析摘要
                        analyzer.print_meta_analysis_summary(meta_result)

                    except Exception as e:
                        logger.error(f"结局指标 {outcome} 分析失败: {str(e)}")
                        failed_outcomes[outcome] = str(e)
                        continue
            else:
                # 如果指定了结局指标，则直接进行分析
                try:
                    meta_result = await analyzer.perform_targeted_meta_analysis(
                        target_paper_ids=paper_ids,
                        outcome_name=outcome_name,
                        query = query,
                        **meta_kwargs
                    )
                    meta_results[outcome_name] = meta_result

                    # 生成森林图
                    if save_plot:
                        if plot_path is None:
                            safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")[:30]
                            safe_outcome = outcome_name.replace(" ", "_").replace("/", "_")[:20]
                            plot_path = f"forest_plot_{safe_query}_{safe_outcome}.png"

                        analyzer.generate_forest_plot(meta_result, plot_path)
                        forest_plots[outcome_name] = plot_path
                        logger.info(f"森林图已保存: {plot_path}")

                    # 打印Meta分析摘要
                    analyzer.print_meta_analysis_summary(meta_result)

                except Exception as e:
                    failed_outcomes[outcome_name] = str(e)
                    logger.error(f"Meta分析失败: {str(e)}")

            # 构建返回结果
            result = {
                "query": query,
                "query_results": query_results,
                "success": len(meta_results) > 0,
                "successful_analyses": len(meta_results),
                "failed_analyses": len(failed_outcomes)
            }

            if meta_results:
                result["meta_analysis_result"] = meta_results
                result["forest_plot_paths"] = forest_plots

            if failed_outcomes:
                result["failed_outcomes"] = failed_outcomes

            # 如果没有任何成功的分析，标记为失败
            if not meta_results:
                result["error"] = "所有结局指标的Meta分析都失败了"

            return result

        except Exception as e:
            logger.error(f"查询和Meta分析过程出错: {str(e)}")
            return {
                "query": query,
                "query_keywords": query_results.get("query_keywords"),
                "query_results": query_results,
                "error": str(e),
                "success": False,
                "meta_analysis_result": meta_results,
                "forest_plot_paths": forest_plots,
                "failed_outcomes": failed_outcomes
            }


if __name__ == "__main__":
    rag = MultiLayerGraphRAG()

    # 使用事件循环运行异步方法
    loop = asyncio.get_event_loop()

    # 检查训练状态
    status = loop.run_until_complete(rag.get_training_status())
    print(f"训练状态: {status}")

    if status["is_trained"]:
        print("检测到已训练的模型，跳过训练阶段")
        print(f"已有 {status['paper_count']} 篇论文，{status['node_count']} 个节点，{status['edge_count']} 条边")
    else:
        # 检查是否有evaluated_papers数据
        evaluated_keys = loop.run_until_complete(rag.meta_graphrag.evaluated_papers_storage.all_keys())

        if evaluated_keys and len(evaluated_keys) > 0:
            # 选项1: 从已有的evaluated_papers.json构建图
            print(f"检测到 {len(evaluated_keys)} 篇已评估论文，从已有数据构建图...")
            result = loop.run_until_complete(rag.build_graph_from_existing_data())
            print(f"构建结果: {result}")
        else:
            # 选项2: 从原始MD文件重新处理
            print("未找到已评估论文，开始从MD文件训练模型...")
            md_dir = "D:/YJS/TMSrag/rTMS-rag/process/dataset/rTMS/markdown/*"
            file_path_list = glob.glob(md_dir + "/*-with-image-refs.md")
            print(f"找到 {len(file_path_list)} 个MD文件")

            for file_path in file_path_list:
                print(f"处理文件: {file_path}")
                with open(file_path, encoding="utf-8") as f:
                    loop.run_until_complete(rag.insert_paper(f.read()))

            print("训练完成！")


    # 直接进行查询和Meta分析

    # 文献1
    # query = "What effect does transcranial magnetic stimulation guided by functional magnetic resonance imaging have on brain activation in patients with depression?(the necessary keyword is fMRI)"
    # 文献2
    # query = "I would like to know the efficacy and long-term maintenance effect of aTMS(including arTMS or aiTBS or aTBS) on patients with MDD"
    # 文献3
    # query = "How effective is TMS for patients with Alzheimer's disease or MCI before 2023?"
    # 文献4
    # query = "How effective is deep transcranial magnetic stimulation for treatment-resistant depression or major depressive disorder before 2024?"
    # 文献5
    # query = "Can TMS improve the gait condition of PD patients with FOG? I want to know the results of FOG-Q, walking time, TUG and UPDRS"
    # 文献6
    # query = "What impact does rTMS combined with cognitive training have on the cognitive function of patients with Alzheimer's disease?"
    # 文献7
    # query = "To evaluate the therapeutic effect of TMS combined with repetitive peripheral magnetic stimulation on upper limb motor dysfunction or upper limb spasticity after stroke(the necessary keyword is peripheral magnetic stimulation)"
    # 文献8
    # query = "How effective is rTMS in patients with central stroke pain (CPSP) or chronic neurogenic pain?"
    # 文献9
    # query = "How effective is low-frequency or high-frequency rTMS in treating cognitive impairment after stroke?(the necessary keyword is stroke and cognitive impairment)"
    # 文献10
    # query = "To evaluate the cognitive function of rTMS in the treatment of patients with PSCI(the necessary keyword is PSCI)"
    # 文献11
    # query = "How effective is tms in treating chronic headaches or chronic migraine or chronic tension-type headache in daily life?"
    # 文献12
    # query = "The therapeutic effect of rTMS on Alzheimer's disease, excluding the influence of cognitive training"
    # 文献13
    # query = "How effective is deep transcranial magnetic stimulation in the treatment of obsessive-compulsive disorder?"
    # 文献14
    # query = "How effective is left high-frequency rTMS in treating major depressive disorder that has failed to use antidepressants twice, excluding artms, atms, aitbs, atbs, drug therapy and deep tms?(the necessary keyword is high frequency)"
    # 文献15
    # query = "Does high-frequency rTMS in DLPFC have a therapeutic effect on the patients with Alzheimer's disease, parkinson or mci?(the necessary keyword is dlpfc)"
    # 文献16
    # query = "Evaluate the therapeutic effect of TMS in managing the condition of ADHD."
    # 文献17
    query = "I would like to know the efficacy and safety of repetitive transcranial magnetic stimulation in cerebellar ataxia"
    # 文献18
    # query = "Are aTMS or rTMS more effective in reducing the severity of obsessive-compulsive disorder on the ybocs scale?"
    # 文献19
    # query = "I would like to know about the efficacy of TMS in alleviating the symptoms of ADHD"
    # 文献20
    # query = "I want to know the efficacy of rTMS in improving the cognition of patients with depression, excluding atms, Alzheimer's disease and dtms(the necessary keyword is cognitive impairment)"
    # 文献21
    # query = "The influence of transcranial magnetic stimulation of the cerebellum on the rehabilitation of stroke, excluding cognitive impairment"
    # 文献22
    # query = "Can rTMS or dtms improve the cognitive function of patients with ADHD, excluding children?"
    # 文献23
    # query = "Can rTMS combined with non-pharmaceutical therapies enhance the antidepressant effect?"
    # 文献24
    # query = "Does rTMS treatment have an improvement effect on the sleep quality of patients?(the necessary keyword is sleep quality)" # with depression, insomnia, parkinson, anxiety disorder or drug dependence
    # 文献25
    # query = "Is TMS combined with drug therapy more effective for adolescents with their first episode of depression?"
    # 文献26
    # query = "What are the effects of TMS on patients with aphasia after stroke?(the necessary keyword is aphasia)"

    # """查询论文并进行Meta分析
    #         args_mode:
    #         community_based
    #         structural
    #         semantic
    #         keyword
    #         comprehensive
    # """

    result = loop.run_until_complete(rag.query_and_meta_analyze(
        query=query,
        effect_type="smd",
        method="auto",
        save_plot=True,
        query_mode="community_based"
    ))

    papers = result["query_results"].get("papers", [])
    top_k25 = papers[:25]
    print("分析结果:", result)
    print("前25篇论文结果:", top_k25)











