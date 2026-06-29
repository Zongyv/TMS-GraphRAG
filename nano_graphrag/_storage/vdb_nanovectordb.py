import asyncio
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB # 一个向量数据库

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    # 向量存储类
    # 余弦相似度阈值, 决定返回结果质量
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        # 初始化向量数据库存储文件和嵌入配置
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        # 初始化向量数据库客户端（NanoVectorDB），并设置嵌入维度
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        # 从全局配置中获取查询的相似度阈值，或使用默认值
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        """
            插入或更新向量数据。
            该方法用于将字典形式的数据插入或更新到向量数据库中。数据首先被转换成适合插入的格式，
            然后分批处理，以避免一次性插入过多数据导致的性能问题。之后，使用异步方式计算各批次数据的嵌入向量，
            并将这些向量附加到数据条目中，最后调用客户端的插入或更新方法完成操作。
            参数:
                data: dict[str, dict] - 一个字典，键是数据的唯一标识，值是包含实际数据内容的字典。
            返回:
                插入或更新操作的结果。
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        # 将数据转换为适合插入的格式
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        # 提取数据中的内容
        contents = [v["content"] for v in data.values()]
        # 分批处理数据
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        # 异步计算嵌入向量
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        # 将嵌入向量附加到数据条目中
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        # 调用NanoVectorDB的upsert方法插入或更新数据
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        """
            根据提供的查询字符串获取最相关的文档。
            此异步方法使用预训练的embedding函数将查询转换为嵌入表示，
            然后在嵌入索引中搜索与查询最相似的文档。
            参数:
                - query: str，用户查询的字符串。
                - top_k: int，返回最相关的文档数量，默认为5。
            返回:
                - 一个列表，包含最相关的文档及其与查询的相似度距离。
        """
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        # 整理结果，添加文档id和距离信息
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()
