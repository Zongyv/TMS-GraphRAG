from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, List, Tuple, Optional, Dict

import numpy as np

from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search 朴素拼接
    naive_max_token_for_text_unit: int = 12000 # 最大token数
    # local search 局部搜索
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33 最大局部内容
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4 最大局部上下文
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27 最大局部报告
    local_community_single_one: bool = False # 是否只使用一个局部社区
    # global search 全局搜索
    global_min_community_rating: float = 0 # 过滤社区报告的最小评分
    global_max_consider_community: float = 512 # 最大考虑社区
    global_max_token_for_community_report: int = 16384 # 最大全局报告tokens数
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

@dataclass
class StudyDetail:
    """单个研究的详细信息"""
    paper_id: str
    doi: str
    title: str
    first_author: str
    year: Optional[str]
    sample_size: int
    intervention_n: int
    control_n: int
    effect_size: float
    se: float
    weight: float
    ci_lower: float
    ci_upper: float

@dataclass
class MetaAnalysisResult:
    """Meta分析结果数据结构"""
    outcome_name: str
    included_studies: int
    total_participants: int
    pooled_effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    heterogeneity_i2: float
    heterogeneity_q: float
    heterogeneity_p: float
    tau_squared: float
    prediction_interval: Optional[Tuple[float, float]] = None
    subgroup_analysis: Optional[Dict] = None
    sensitivity_analysis: Optional[Dict] = None
    publication_bias: Optional[Dict] = None
    study_details: Optional[List[StudyDetail]] = None  # 添加研究详细信息
    effect_type: str = "smd"  # 效应量类型
    method: str = "random"  # Meta分析方法

# 文本块
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

# 单个社群
SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[list[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)

# 添加了字符串格式以及json格式报告的社区class
class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")

# 存储空间
@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass

# 向量存储基类
@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

# 键值存储基类
@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError

# 图存储基类
@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError
    
    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")
