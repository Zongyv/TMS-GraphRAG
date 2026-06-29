import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List
import networkx as nx # 图数据库
import numpy as np
import asyncio

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    # 加载并返回一个NetworkX图，存储格式为graphml
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            try:
                # 检查文件是否为空
                if os.path.getsize(file_name) == 0:
                    logger.warning(f"GraphML文件为空: {file_name}")
                    return None
                
                # 尝试读取GraphML文件
                graph = nx.read_graphml(file_name)
                
                # 反序列化节点属性中的JSON字符串
                for node_id, node_data in graph.nodes(data=True):
                    for key, value in list(node_data.items()):
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                node_data[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                pass  # 保持原始字符串值
                
                # 反序列化边属性中的JSON字符串
                for source, target, edge_data in graph.edges(data=True):
                    for key, value in list(edge_data.items()):
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                edge_data[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                pass  # 保持原始字符串值
                
                return graph
                
            except Exception as e:
                logger.warning(f"读取GraphML文件失败: {file_name}, 错误: {e}")
                # 备份损坏的文件
                backup_name = f"{file_name}.backup"
                try:
                    os.rename(file_name, backup_name)
                    logger.info(f"已将损坏的文件备份为: {backup_name}")
                except:
                    pass
                return None
        return None

    # 将一个NetworkX图写入到文件中，存储格式为graphml
    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        
        # 创建图的副本以避免修改原图
        graph_copy = graph.copy()
        
        # 序列化节点属性中的复杂数据类型
        for node_id, node_data in graph_copy.nodes(data=True):
            for key, value in list(node_data.items()):
                if isinstance(value, (dict, list, tuple, set)):
                    node_data[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, np.ndarray):
                    node_data[key] = json.dumps(value.tolist(), ensure_ascii=False)
                elif not isinstance(value, (str, int, float, bool)):
                    node_data[key] = str(value)
        
        # 序列化边属性中的复杂数据类型
        for source, target, edge_data in graph_copy.edges(data=True):
            for key, value in list(edge_data.items()):
                if isinstance(value, (dict, list, tuple, set)):
                    edge_data[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, np.ndarray):
                    edge_data[key] = json.dumps(value.tolist(), ensure_ascii=False)
                elif not isinstance(value, (str, int, float, bool)):
                    edge_data[key] = str(value)
        
        nx.write_graphml(graph_copy, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        """
            返回图的最大连通分量，并以稳定的方式排序节点和边。
                参数:
                    graph (nx.Graph): 输入的 NetworkX 图。
                返回:
                    nx.Graph: 输入图的最大连通分量，以稳定方式排序。
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        # 强制转换为nx.Graph类型
        graph = cast(nx.Graph, largest_connected_component(graph))
        # 将节点名称去除空白字符
        node_mapping = {node: html.unescape(node.strip()) for node in graph.nodes()}  # type: ignore
        # 重新标记节点
        graph = nx.relabel_nodes(graph, node_mapping)
        # 稳定化图
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        """
            确保无向图以相同的关系读取时始终相同。
            
                参数:
                    graph (nx.Graph): 输入的网络图。
                
                返回:
                    nx.Graph: 经过稳定处理的网络图。
        """
        # 根据输入图的类型初始化一个新的图实例
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        # 对节点进行排序，以确保节点的添加顺序一致
        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
        # 向新图中添加排序后的节点
        fixed_graph.add_nodes_from(sorted_nodes)
        # 将边数据存储到列表中以便后续处理
        edges = list(graph.edges(data=True))

        # 如果图不是有向图，则对边进行排序，以确保边的顺序一致
        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                # 使用字符串的字典序比较，确保较小的节点ID在前
                if str(source) > str(target):
                    return target, source, edge_data
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        # 定义获取边的键的函数，用于后续边的排序
        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"
        # 对边进行排序
        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """
            初始化函数，用于加载图数据并初始化相关属性。
            该函数首先根据全局配置中的工作目录和实例的命名空间来确定graphml文件的路径。
            然后尝试从该路径加载已存在的图数据。如果图数据存在，则使用NetworkXStorage加载，
            并记录日志信息包括图的节点数和边数。如果图数据不存在，则初始化一个新的无向图。
            最后，初始化两个算法字典，分别用于图的聚类算法和节点嵌入算法。
        """
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        """
            异步检查图中是否存在指定节点。
            该方法主要用于确定图结构中是否包含特定的节点。它通过调用底层图对象的has_node方法，
            以高效的方式查询节点是否存在。
                参数:
                    node_id (str): 要检查的节点的唯一标识符。
                返回:
                    bool: 如果图中存在该节点，则返回True，否则返回False。
        """
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)
    
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        return await asyncio.gather(*[self.get_node(node_id) for node_id in node_ids])

    # 获取指定节点的度数
    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        return await asyncio.gather(*[self.node_degree(node_id) for node_id in node_ids])

    # 计算两个节点的度数之和
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        return await asyncio.gather(*[self.edge_degree(src_id, tgt_id) for src_id, tgt_id in edge_pairs])

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        return await asyncio.gather(*[self.get_edge(source_node_id, target_node_id) for source_node_id, target_node_id in edge_pairs])

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        return await asyncio.gather(*[self.get_node_edges(node_id) for node_id
        in node_ids])

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        await asyncio.gather(*[self.upsert_node(node_id, node_data) for node_id, node_data in nodes_data])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        await asyncio.gather(*[self.upsert_edge(source_node_id, target_node_id, edge_data) 
                for source_node_id, target_node_id, edge_data in edges_data])
    
    # 根据指定的算法执行聚类操作。
    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        
        for node_id, node_data in self._graph.nodes(data=True):
            # 检查节点是否有clusters属性
            if "clusters" not in node_data:
                logger.debug(f"节点 {node_id} 没有clusters属性，跳过")
                continue
                
            try:
                clusters = json.loads(node_data["clusters"])
                this_node_edges = self._graph.edges(node_id)

                for cluster in clusters:
                    level = cluster["level"]
                    cluster_key = str(cluster["cluster"])
                    levels[level].add(cluster_key)
                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [tuple(sorted([str(e[0]), str(e[1])])) for e in this_node_edges]
                    )
                    
                    # 处理chunk_ids - 适配不同的节点结构
                    chunk_ids = []
                    if "source_id" in node_data:
                        # 原始GraphRAG格式
                        source_id = node_data["source_id"]
                        if isinstance(source_id, str):
                            chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                        else:
                            chunk_ids = [str(source_id)]
                    elif "paper_id" in node_data:
                        # 论文节点格式
                        chunk_ids = [node_data["paper_id"]]
                    elif "id" in node_data:
                        # 通用ID格式
                        chunk_ids = [node_data["id"]]
                    else:
                        # 使用节点ID作为fallback
                        chunk_ids = [str(node_id)]
                    
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))
                        
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"处理节点 {node_id} 的clusters数据时出错: {e}")
                continue

        # 计算子社区关系
        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        # 转换集合为列表并计算occurrence
        for k, v in results.items():
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max(max_num_ids, 1) if max_num_ids > 0 else 0.0
        
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        """将聚类数据添加到图节点中"""
        for node_id, clusters in cluster_data.items():
            if self._graph.has_node(node_id):
                # 确保节点存在，然后添加clusters属性
                self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)
            else:
                logger.warning(f"节点 {node_id} 不存在于图中，跳过添加clusters属性")

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
