import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        # 工作目录
        working_dir = self.global_config["working_dir"]
        # 根据命名空间生成特定的 JSON 文件名，用于存储键值数据
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        # 加载存储的数据，如果文件不存在或为空，则初始化为空字典
        self._data = load_json(self._file_name) or {}
        # 打印日志，显示加载的数据条数
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    # 获取所有的键列表
    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    # 索引操作完成后，将当前数据写入 JSON 文件
    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    # 根据给定的 ID 获取对应的键值数据
    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        """
            根据ID列表获取数据项。
            参数:
                ids (list): 需要获取数据的ID列表。
                fields (list, 可选): 限制返回数据中的字段。如果未提供，默认为None，将返回完整数据项。
            返回:
                list: 包含按指定ID列表顺序排列的数据项的列表。如果某些ID未找到数据项，则相应位置为None。
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                # 如果数据项存在，并且ID在_data字典中，则构建一个仅包含fields中字段的新字典
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    # 过滤数据项# 过滤出不在数据存储中的键列表
    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    # 更新数据项
    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    # 删除数据项
    async def drop(self):
        self._data = {}

    async def delete(self, id: str) -> bool:
        """删除指定ID的数据"""
        if id in self._data:
            del self._data[id]
            logger.info(f"已删除 {self.namespace} 中的数据: {id}")
            return True
        return False
