import glob
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from nano_graphrag._utils import logger
from nano_graphrag.base import MetaAnalysisResult, StudyDetail
from nano_graphrag.meta_analysis_graphrag import MetaAnalysisGraphRAG
import asyncio
from nano_graphrag._utils import always_get_an_event_loop
from nano_graphrag.prompt import PROMPTS
from outcome_aliases_config import OUTCOME_ALIASES
from test import (
    resilient_model_call,
    resilient_cheap_model_call
)


class MetaAnalyzer:
    """Meta分析核心类"""
    
    def __init__(self, graphrag_instance: MetaAnalysisGraphRAG):
        self.graphrag = graphrag_instance
        self.papers_data = {}
        self.best_model_func = resilient_model_call
        self.cheap_model_func = resilient_cheap_model_call
        # 不在__init__中同步加载数据
    
    async def load_papers_data(self):
        """从GraphRAG实例加载论文数据"""
        try:
            # 先获取所有键
            all_keys = await self.graphrag.evaluated_papers_storage.all_keys()
            
            # 然后根据键获取所有数据
            all_papers_list = await self.graphrag.evaluated_papers_storage.get_by_ids(all_keys)
            
            # 将列表转换为字典格式
            self.papers_data = {}
            for i, paper_data in enumerate(all_papers_list):
                if paper_data is not None:
                    self.papers_data[all_keys[i]] = paper_data
            
            logger.info(f"加载了 {len(self.papers_data)} 篇论文数据")
        except Exception as e:
            logger.error(f"加载论文数据失败: {str(e)}")
            self.papers_data = {}
    
    def get_available_outcomes(self) -> Dict[str, List[str]]:
        """获取所有可用的结局指标"""
        outcomes = {"primary": [], "secondary": []}
        
        for paper_id, paper_data in self.papers_data.items():
            # 主要结局指标
            for outcome in paper_data.get("primary_outcomes", []):
                outcome_name = outcome.get("outcome_name") or ""
                if outcome_name and outcome_name not in outcomes["primary"]:
                    outcomes["primary"].append(outcome_name)
            
            # 次要结局指标
            for outcome in paper_data.get("secondary_outcomes", []):
                outcome_name = outcome.get("outcome_name") or ""
                if outcome_name and outcome_name not in outcomes["secondary"]:
                    outcomes["secondary"].append(outcome_name)
        
        return outcomes

    def analyze_data_completeness(self):
        """分析数据完整性"""
        print("\n=== 数据完整性分析 ===")

        total_papers = len(self.papers_data)
        papers_with_outcomes = 0
        papers_with_complete_data = 0
        papers_with_valid_timepoints = 0

        outcome_counts = {}
        baseline_only_outcomes = {}

        for paper_id, paper_data in self.papers_data.items():
            title = paper_data.get("title")[:50] or "Unknown"
            print(f"\n论文: {title}...")
            print(f"  研究类型: {paper_data.get('study_type') or 'Unknown'}")
            print(f"  样本量: {paper_data.get('sample_size') or 'Unknown'}")



            has_outcomes = False
            has_complete = False
            has_valid_timepoints = False

            # 检查主要结局指标
            primary_outcomes = paper_data.get("primary_outcomes", [])
            if primary_outcomes:
                has_outcomes = True
                print(f"  主要结局指标 ({len(primary_outcomes)}个):")
                for outcome in primary_outcomes:
                    outcome_name = outcome.get("outcome_name") or "Unknown"
                    time_point = outcome.get("time_point") or ""

                    print(f"    - {outcome_name} (时间点: {time_point})")

                    # 检查时间点是否仅为baseline
                    if self._is_baseline_only(time_point):
                        print(f"{outcome_name}仅有baseline数据，跳过")
                        if outcome_name not in baseline_only_outcomes:
                            baseline_only_outcomes[outcome_name] = 0
                        baseline_only_outcomes[outcome_name] += 1
                        continue

                    # 统计有效结局指标出现次数
                    if outcome_name not in outcome_counts:
                        outcome_counts[outcome_name] = 0
                    outcome_counts[outcome_name] += 1
                    has_valid_timepoints = True

                    # 检查数据完整性
                    if self._is_data_complete(outcome):
                        has_complete = True
                        print(f"      ✓ 数据完整")
                    else:
                        print(f"      ✗ 数据不完整")

            # 检查次要结局指标
            secondary_outcomes = paper_data.get("secondary_outcomes", [])
            if secondary_outcomes:
                has_outcomes = True
                print(f"  次要结局指标 ({len(secondary_outcomes)}个):")
                for outcome in secondary_outcomes:
                    outcome_name = outcome.get("outcome_name") or "Unknown"
                    time_point = outcome.get("time_point") or ""

                    print(f"    - {outcome_name} (时间点: {time_point})")

                    # 检查时间点是否仅为baseline
                    if self._is_baseline_only(time_point):
                        print(f"{outcome_name}仅有baseline数据，跳过")
                        if outcome_name not in baseline_only_outcomes:
                            baseline_only_outcomes[outcome_name] = 0
                        baseline_only_outcomes[outcome_name] += 1
                        continue

                    # 统计有效结局指标出现次数
                    if outcome_name not in outcome_counts:
                        outcome_counts[outcome_name] = 0
                    outcome_counts[outcome_name] += 1
                    has_valid_timepoints = True

                    # 检查数据完整性
                    if self._is_data_complete(outcome):
                        has_complete = True
                        print(f"      ✓ 数据完整")
                    else:
                        print(f"      ✗ 数据不完整")

            if not has_outcomes:
                print("  没有提取到结局指标")
            elif not has_valid_timepoints:
                print("  所有结局指标都仅有baseline数据")

            if has_outcomes:
                papers_with_outcomes += 1
            if has_complete:
                papers_with_complete_data += 1
            if has_valid_timepoints:
                papers_with_valid_timepoints += 1

        print(f"\n=== 总体统计 ===")
        print(f"总论文数: {total_papers}")
        print(f"有结局指标的论文: {papers_with_outcomes}")
        print(f"有有效时间点的论文: {papers_with_valid_timepoints}")
        print(f"有完整数据的论文: {papers_with_complete_data}")

        print(f"\n=== 有效结局指标统计 ===")
        sorted_outcomes = sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True)
        for outcome, count in sorted_outcomes:
            print(f"{outcome}: {count}篇论文")

        if baseline_only_outcomes:
            print(f"\n=== 仅有baseline数据的结局指标 ===")
            sorted_baseline = sorted(baseline_only_outcomes.items(), key=lambda x: x[1], reverse=True)
            for outcome, count in sorted_baseline:
                print(f"{outcome}: {count}篇论文 (已排除)")

        return outcome_counts

    
    async def filter_papers_for_meta_analysis(
        self,
        outcome_name: str,
        query: str = "",
        study_types: List[str] = ["RCT", "multi-arm RCT", "open label", "case report"],
        min_sample_size: int = 5,
        max_bias_risk: str = "High",
        require_complete_data: bool = True,
        timepoint_selection: str = "auto"
    ) -> List[Dict]:
        """筛选适合Meta分析的论文"""

        print(f"\n=== 筛选论文进行 {outcome_name} 的Meta分析 ===")
        print(f"用户查询: {query}")
        print(f"时间点选择策略: {timepoint_selection}")
        print(f"筛选条件:")
        print(f"  - 研究类型: {study_types}")
        print(f"  - 最小样本量: {min_sample_size}")
        print(f"  - 最大偏倚风险: {max_bias_risk}")
        print(f"  - 要求完整数据: {require_complete_data}")
        
        filtered_papers = []
        risk_hierarchy = {"Low": 1, "Some concerns": 2, "High": 3}
        max_risk_score = risk_hierarchy.get(max_bias_risk, 3)
        
        for paper_id, paper_data in self.papers_data.items():
            title = paper_data.get("title")[:50] or "Unknown"
            print(f"\n检查论文: {title}...")
            
            # 基本筛选条件
            study_type = paper_data.get("study_type")
            if not self._study_type_matches(study_type, study_types):
                print(f"研究类型不符: '{study_type}' 不在 {study_types} 中")
                continue

            sample_size = paper_data.get("sample_size", 0)
            if sample_size and sample_size < min_sample_size:
                print(f"样本量不足: {sample_size} < {min_sample_size}")
                continue
            elif not sample_size:
                print(f"样本量未知，跳过样本量检查")
            
            # RoB2偏倚风险筛选
            rob2_data = paper_data.get("rob2_assessment", {})
            overall_risk = rob2_data.get("overall_bias_risk", {}).get("risk_level", "High")
            paper_risk_score = risk_hierarchy.get(overall_risk, 3)
            
            if paper_risk_score > max_risk_score:
                print(f"偏倚风险过高: {overall_risk}")
                continue
            
            # 检查是否有目标结局指标的数据（异步调用，传入query）
            outcome_data = await self._extract_outcome_data(paper_data, outcome_name, query, timepoint_selection)
            if not outcome_data:
                self._log_available_outcomes(paper_data)
                print(f"没有找到结局指标: {outcome_name}")
                continue
            
            # 检查数据完整性
            if require_complete_data and not self._is_data_complete(outcome_data):
                print(f"结局指标数据不完整")
                self._log_missing_data(outcome_data)
                continue
            
            print(f"论文符合条件")
            print(f"    结局指标: {outcome_data.get('outcome_name', 'Unknown')}")
            print(f"    干预组: n={outcome_data.get('intervention_group', {}).get('n', '?')}")
            print(f"    对照组: n={outcome_data.get('control_group', {}).get('n', '?')}")
            
            filtered_papers.append({
                "paper_id": paper_id,
                "paper_data": paper_data,
                "outcome_data": outcome_data
            })
        
        print(f"\n=== 筛选完成：{len(filtered_papers)} 篇论文符合条件 ===")
        return filtered_papers
    
    def _study_type_matches(self, study_type: str, target_types: List[str]) -> bool:
        """检查研究类型是否匹配"""
        if not study_type:
            return False
        
        study_type_clean = study_type.lower()
        
        for target in target_types:
            target_clean = target.lower()
            
            # 精确匹配
            if study_type_clean == target_clean:
                return True
            
            # RCT的各种表达方式
            if target_clean == "rct":
                rct_patterns = [
                    "randomized controlled trial",
                    "randomised controlled trial", 
                    "randomized",
                    "randomised",
                    "rct"
                ]
                if any(pattern in study_type_clean for pattern in rct_patterns):
                    return True
        
        return False
    
    def _log_available_outcomes(self, paper_data: Dict):
        """记录论文中可用的结局指标"""
        all_outcomes = []
        
        for outcome in paper_data.get("primary_outcomes", []):
            outcome_name = outcome.get("outcome_name") or ""
            if outcome_name:
                all_outcomes.append(f"主要: {outcome_name}")
        
        for outcome in paper_data.get("secondary_outcomes", []):
            outcome_name = outcome.get("outcome_name") or ""
            if outcome_name:
                all_outcomes.append(f"次要: {outcome_name}")
        
        if all_outcomes:
            print(f"论文中的结局指标: {'; '.join(all_outcomes)}")
        else:
            print(f"论文中没有提取到结局指标")
    
    def _log_missing_data(self, outcome_data: Dict):
        """记录缺失的数据"""
        intervention = outcome_data.get("intervention_group", {})
        control = outcome_data.get("control_group", {})
        
        missing = []
        for group_name, group_data in [("干预组", intervention), ("对照组", control)]:
            for field in ["n", "mean", "sd"]:
                if not group_data.get(field):
                    missing.append(f"{group_name}.{field}")
        
        if missing:
            print(f"    缺失数据: {', '.join(missing)}")


    async def _select_best_timepoint_with_llm(self, valid_outcomes: List[Dict], query: str = "", strategy: str = "auto") -> Dict:
        """使用LLM选择最佳时间点的结局数据"""

        if len(valid_outcomes) <= 1:
            return valid_outcomes[0] if valid_outcomes else None

        # 非LLM策略：根据策略参数直接选择
        if strategy == "earliest":
            print(f"    策略'{strategy}': 选择第一个治疗后时间点")
            return valid_outcomes[0]
        elif strategy == "latest":
            print(f"    策略'{strategy}': 选择最后一个时间点")
            return valid_outcomes[-1]

        # 构建时间点信息
        timepoint_descriptions = []
        for i, outcome in enumerate(valid_outcomes):
            tp = outcome.get('time_point') or '未知'
            # 补充组信息以便LLM更准确判断
            ig = outcome.get("intervention_group", {})
            cg = outcome.get("control_group", {})
            detail = f"选项{i}: {tp}"
            if ig.get("n") and cg.get("n"):
                detail += f" (干预组n={ig['n']}, 对照组n={cg['n']})"
            timepoint_descriptions.append(detail)

        query_context = f"用户研究问题: {query}\n" if query else "用户未提供具体问题，请依据默认原则选择。"
        timepoint_prompt = PROMPTS["timepoint_prompt"].format(
            query_context=query_context,
            timepoint_descriptions="\n".join(timepoint_descriptions)
        )

        try:
            response = await self.best_model_func(timepoint_prompt)
            selected_index = int(response.strip())

            if 0 <= selected_index < len(valid_outcomes):
                selected_tp = valid_outcomes[selected_index].get('time_point') or '未知'
                print(f"    LLM基于查询选择时间点: {selected_tp}")
                return valid_outcomes[selected_index]
            else:
                print(f"    LLM返回无效索引 {selected_index}，按'{strategy}'策略回退")
                return self._fallback_timepoint(valid_outcomes, strategy)

        except Exception as e:
            print(f"    LLM选择时间点失败: {e}，按'{strategy}'策略回退")
            return self._fallback_timepoint(valid_outcomes, strategy)

    def _fallback_timepoint(self, valid_outcomes: List[Dict], strategy: str = "auto") -> Dict:
        """当LLM选择失败时，根据策略回退选择时间点"""
        if strategy == "latest":
            print(f"    回退: 选择最后一个时间点")
            return valid_outcomes[-1]
        # 默认策略(auto/earliest)：选择第一个治疗后时间点（post-treatment优先）
        print(f"    回退: 选择第一个治疗后时间点")
        return valid_outcomes[0]


    async def _extract_outcome_data(self, paper_data: Dict, outcome_name: str, query: str = "", timepoint_selection: str = "auto") -> Optional[Dict]:
        """从论文数据中提取特定结局指标的数据（使用LLM优选时间点）"""

        def get_valid_outcomes(outcomes_list):
            """获取所有有效的结局数据"""
            valid_outcomes = []

            for outcome in outcomes_list:
                if self._outcome_name_match_with_aliases(outcome.get("outcome_name", ""), outcome_name):
                    # 跳过仅有baseline的数据
                    if self._is_baseline_only(outcome.get("time_point") or ""):
                        continue
                    valid_outcomes.append(outcome)

            return valid_outcomes

        # 检查主要结局指标
        primary_outcomes = get_valid_outcomes(paper_data.get("primary_outcomes", []))
        if primary_outcomes:
            if len(primary_outcomes) == 1:
                outcome_data = primary_outcomes[0]
            else:
                # 使用LLM选择最佳时间点（传入query和策略）
                outcome_data = await self._select_best_timepoint_with_llm(primary_outcomes, query, timepoint_selection)

            # 检查是否为多臂RCT数据结构
            if outcome_data and outcome_data.get("groups"):
                return await self._convert_multi_arm_to_two_arm(outcome_data, paper_data, query)
            # 检查是否为交叉实验数据结构
            elif outcome_data and outcome_data.get("periods"):
                return self._convert_crossover_to_two_arm(outcome_data)
            return outcome_data

        # 检查次要结局指标
        secondary_outcomes = get_valid_outcomes(paper_data.get("secondary_outcomes", []))
        if secondary_outcomes:
            if len(secondary_outcomes) == 1:
                outcome_data = secondary_outcomes[0]
            else:
                # 使用LLM选择最佳时间点（传入query和策略）
                outcome_data = await self._select_best_timepoint_with_llm(secondary_outcomes, query, timepoint_selection)

            # 检查是否为多臂RCT数据结构
            if outcome_data and outcome_data.get("groups"):
                return await self._convert_multi_arm_to_two_arm(outcome_data, paper_data, query)
            # 检查是否为交叉实验数据结构
            elif outcome_data and outcome_data.get("periods"):
                return self._convert_crossover_to_two_arm(outcome_data)
            return outcome_data

        return None

    def _convert_crossover_to_two_arm(self, outcome_data: Dict) -> Optional[Dict]:
        """将交叉实验数据转换为双臂数据结构（仅使用第一期数据）"""

        periods = outcome_data.get("periods", [])
        if not periods:
            print(f"    交叉实验没有时期数据")
            return None

        # 仅使用第一期数据
        first_period = periods[0]
        period_name = first_period.get("period_name", "Period 1")
        conditions = first_period.get("conditions", [])

        if len(conditions) < 2:
            print(f"    第一期条件数不足: {len(conditions)}")
            return None

        print(f"    使用交叉实验第一期数据: {period_name}")

        # 根据condition_type识别干预组和对照组
        intervention_condition = None
        control_condition = None

        for condition in conditions:
            condition_type = condition.get("condition_type", "").lower()

            if condition_type == "intervention":
                intervention_condition = condition
            elif condition_type == "control":
                control_condition = condition

        # 检查是否成功识别
        if not intervention_condition or not control_condition:
            print(f"    无法识别干预组和对照组")
            return None

        # 转换为标准的双臂数据结构
        converted_data = {
            "outcome_name": outcome_data.get("outcome_name"),
            "time_point": outcome_data.get("time_point"),
            "intervention_group": {
                "n": intervention_condition.get("n"),
                "mean": intervention_condition.get("mean"),
                "sd": intervention_condition.get("sd"),
                "group_name": f"{intervention_condition.get('sequence_group', '')} - intervention"
            },
            "control_group": {
                "n": control_condition.get("n"),
                "mean": control_condition.get("mean"),
                "sd": control_condition.get("sd"),
                "group_name": f"{control_condition.get('sequence_group', '')} - control"
            },
            "original_structure": "crossover",
            "period_used": period_name,
            "total_periods": len(periods)
        }

        print(f"    交叉实验转换: intervention vs control (第一期)")

        return converted_data

    async def _convert_multi_arm_to_two_arm(self, outcome_data: Dict, paper_data: Dict, query: str = "") -> Optional[Dict]:
        """将多臂RCT数据转换为双臂数据结构（使用LLM根据用户问题选择最相关的两组）"""
        
        groups = outcome_data.get("groups", [])
        if len(groups) < 2:
            print(f"    多臂RCT组数不足: {len(groups)}")
            return None
        
        # 使用LLM根据用户问题选择最相关的两组
        selected_groups = await self._select_two_groups_with_llm(groups, paper_data, query)
        
        if not selected_groups or len(selected_groups) != 2:
            print(f"    LLM未能选择有效的两组")
            return None
        
        intervention_group, control_group = selected_groups
        
        # 转换为标准的双臂数据结构
        converted_data = {
            "outcome_name": outcome_data.get("outcome_name"),
            "time_point": outcome_data.get("time_point"),
            "intervention_group": {
                "n": intervention_group.get("n"),
                "mean": intervention_group.get("mean"),
                "sd": intervention_group.get("sd"),
                "group_name": intervention_group.get("group_name"),
                "group_id": intervention_group.get("group_id")
            },
            "control_group": {
                "n": control_group.get("n"),
                "mean": control_group.get("mean"),
                "sd": control_group.get("sd"),
                "group_name": control_group.get("group_name"),
                "group_id": control_group.get("group_id")
            },
            "original_structure": "multi_arm",
            "total_groups": len(groups)
        }
        
        print(f"    多臂RCT转换: {intervention_group.get('group_name')} vs {control_group.get('group_name')}")
        
        return converted_data

    async def _select_two_groups_with_llm(self, groups: List[Dict], paper_data: Dict, query: str = "") -> Optional[List[Dict]]:
        """使用LLM从多个组中根据用户问题选择最相关的两组进行比较"""
        
        # 构建组信息描述
        group_descriptions = []
        for i, group in enumerate(groups):
            group_name = group.get("group_name", f"Group {i+1}")
            group_id = group.get("group_id", i+1)
            n = group.get("n", "unknown")
            mean = group.get("mean", "unknown")
            sd = group.get("sd", "unknown")
            
            description = f"组{group_id} ({group_name}): n={n}, mean={mean}, sd={sd}"
            group_descriptions.append(description)
        
        # 获取TMS参数信息（如果有）
        tms_info = ""
        tms_metadata = paper_data.get("tms_metadata", {})
        if tms_metadata.get("intervention_groups"):
            tms_info = "\n\nTMS刺激参数信息:\n"
            for tms_group in tms_metadata["intervention_groups"]:
                group_id = tms_group.get("group_id")
                group_name = tms_group.get("group_name")
                frequency = tms_group.get("stimulation_frequency", "unknown")
                intensity = tms_group.get("stimulation_intensity", "unknown")
                target = tms_group.get("brain_target", "unknown")
                tms_info += f"组{group_id} ({group_name}): 频率={frequency}, 强度={intensity}, 靶点={target}\n"
        
        # 构建用户问题上下文
        query_context = f"\n\n用户研究问题: {query}\n" if query else ""
        
        group_selection_prompt = PROMPTS["group_selection_prompt"].format(
            query_context=query_context,
            group_descriptions="\n".join(group_descriptions),
            tms_info=tms_info,
            total_groups=len(groups)
        )
        
        try:
            response = await self.best_model_func(group_selection_prompt)
            result = json.loads(response.strip())
            
            intervention_id = result.get("intervention_group_id")
            control_id = result.get("control_group_id")
            rationale = result.get("rationale", "")
            
            print(f"    LLM选择组别: 干预组ID={intervention_id}, 对照组ID={control_id}")
            print(f"    选择理由: {rationale}")
            
            # 根据group_id找到对应的组
            intervention_group = next((g for g in groups if g.get("group_id") == intervention_id), None)
            control_group = next((g for g in groups if g.get("group_id") == control_id), None)
            
            if intervention_group and control_group:
                return [intervention_group, control_group]
            else:
                print(f"    未找到对应的组: intervention_id={intervention_id}, control_id={control_id}")
                return None
            
        except Exception as e:
            print(f"    LLM选择组别失败: {e}")
            # 降级策略：选择第一个干预组和对照组
            return self._fallback_group_selection(groups)

    def _fallback_group_selection(self, groups: List[Dict]) -> Optional[List[Dict]]:
        """降级策略：当LLM失败时，使用启发式规则选择组别"""
        
        # 尝试识别对照组（包含sham、placebo、control等关键词）
        control_keywords = ["sham", "placebo", "control", "假刺激", "对照"]
        intervention_groups = []
        control_groups = []
        
        for group in groups:
            group_name = group.get("group_name", "").lower()
            if any(keyword in group_name for keyword in control_keywords):
                control_groups.append(group)
            else:
                intervention_groups.append(group)
        
        # 如果找到了对照组和干预组
        if intervention_groups and control_groups:
            # 选择第一个干预组和第一个对照组
            print(f"    降级策略: 选择 {intervention_groups[0].get('group_name')} vs {control_groups[0].get('group_name')}")
            return [intervention_groups[0], control_groups[0]]
        
        # 如果没有明确的对照组，选择前两组
        if len(groups) >= 2:
            print(f"    降级策略: 选择前两组")
            return [groups[0], groups[1]]
        
        return None

    def _is_baseline_only(self, time_point: str) -> bool:
        """检查时间点是否仅为baseline"""
        if not time_point:
            return False

        time_point_lower = time_point.lower().strip()

        # baseline关键词
        baseline_keywords = [
            "baseline", "base line", "pre-treatment", "pretreatment",
            "pre-intervention", "preintervention", "before treatment",
        ]

        # 检查是否只包含baseline关键词
        for keyword in baseline_keywords:
            if keyword in time_point_lower:
                # 检查是否同时包含其他时间点
                post_keywords = [
                    "post", "after", "follow", "week", "month", "day",
                ]

                # 如果同时包含post关键词，说明不是仅baseline
                if any(post_kw in time_point_lower for post_kw in post_keywords):
                    return False

                # 如果只有baseline关键词，返回True
                return True

        return False

    def _outcome_name_match_with_aliases(self, paper_outcome: str, target_outcome: str) -> bool:
        """使用别名配置进行结局指标匹配"""
        if not paper_outcome or not target_outcome:
            return False

        # print(f"  匹配检查开始: '{paper_outcome}' vs '{target_outcome}'")

        # 获取两个指标的标准名称
        paper_standard = self._get_standard_outcome_name(paper_outcome)
        target_standard = self._get_standard_outcome_name(target_outcome)

        # print(f"  标准名称: '{paper_standard}' vs '{target_standard}'")

        # 比较标准名称
        result = paper_standard.lower() == target_standard.lower()
        # print(f"  匹配结果: {result}")

        return result
    
    def _is_data_complete(self, outcome_data: Dict) -> bool:
        """检查结局数据是否完整"""
        intervention_group = outcome_data.get("intervention_group", {})
        control_group = outcome_data.get("control_group", {})
        
        required_fields = ["n", "mean", "sd"]
        
        for field in required_fields:
            if (not intervention_group.get(field) or 
                not control_group.get(field)):
                return False
        
        return True

    def calculate_effect_size(self, outcome_data: Dict, effect_type: str = "smd") -> Tuple[float, float]:
        """计算效应量和标准误"""

        try:
            intervention = outcome_data["intervention_group"]
            control = outcome_data["control_group"]

            # 检查必需字段是否存在且不为None
            required_fields = ["n", "mean", "sd"]
            for group_name, group_data in [("intervention", intervention), ("control", control)]:
                for field in required_fields:
                    if field not in group_data or group_data[field] is None:
                        raise ValueError(f"{group_name}组缺少必需字段: {field}")

            n1, mean1, sd1 = intervention["n"], intervention["mean"], intervention["sd"]
            n2, mean2, sd2 = control["n"], control["mean"], control["sd"]

            # 检查数值类型
            try:
                n1, n2 = int(n1), int(n2)
                mean1, mean2, sd1, sd2 = float(mean1), float(mean2), float(sd1), float(sd2)
            except (ValueError, TypeError) as e:
                raise ValueError(f"数值转换失败: {e}")

            if effect_type == "smd":  # 标准化均数差 (Hedges' g)
                return self._calculate_hedges_g(n1, mean1, sd1, n2, mean2, sd2)
            elif effect_type == "md":  # 均数差
                return self._calculate_mean_difference(n1, mean1, sd1, n2, mean2, sd2)
            else:
                raise ValueError(f"不支持的效应量类型: {effect_type}")

        except Exception as e:
            # 打印详细的调试信息
            print(f"计算效应量时出错: {e}")
            print(f"outcome_data结构: {outcome_data}")
            if "intervention_group" in outcome_data:
                print(f"干预组数据: {outcome_data['intervention_group']}")
            if "control_group" in outcome_data:
                print(f"对照组数据: {outcome_data['control_group']}")
            raise
    
    def _calculate_hedges_g(self, n1: int, mean1: float, sd1: float, 
                           n2: int, mean2: float, sd2: float) -> Tuple[float, float]:
        """计算Hedges' g和标准误"""
        
        # 合并标准差
        pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_sd
        
        # 小样本校正因子
        j = 1 - (3 / (4 * (n1 + n2) - 9))
        
        # Hedges' g
        hedges_g = cohens_d * j
        
        # 标准误
        se = np.sqrt((n1 + n2) / (n1 * n2) + hedges_g**2 / (2 * (n1 + n2 - 2)))
        
        return hedges_g, se
    
    def _calculate_mean_difference(self, n1: int, mean1: float, sd1: float,
                                  n2: int, mean2: float, sd2: float) -> Tuple[float, float]:
        """计算均数差和标准误"""
        
        md = mean1 - mean2
        se = np.sqrt(sd1**2/n1 + sd2**2/n2)
        
        return md, se
    
    async def perform_meta_analysis(
        self,
        outcome_name: str,
        query: str = "",
        effect_type: str = "md",
        method: str = "auto",
        i2_threshold: float = 50.0,
        heterogeneity_p_threshold: float = 0.05,
        timepoint_selection: str = "auto",
        **filter_kwargs
    ) -> MetaAnalysisResult:
        """执行Meta分析"""

        # 筛选论文（传入query和timepoint_selection）
        filtered_papers = await self.filter_papers_for_meta_analysis(
            outcome_name,
            query=query,
            timepoint_selection=timepoint_selection,
            **filter_kwargs
        )
        
        if len(filtered_papers) < 2:
            raise ValueError(f"筛选出的论文数量不足 ({len(filtered_papers)} < 2)")
        
        # 计算各研究的效应量
        effect_sizes = []
        standard_errors = []
        weights = []
        study_details = []
        
        print(f"\n=== 计算效应量 ===")
        for i, paper in enumerate(filtered_papers):
            outcome_data = paper["outcome_data"]
            paper_data = paper["paper_data"]
            
            try:
                effect_size, se = self.calculate_effect_size(outcome_data, effect_type)
                weight = 1 / (se ** 2)
                
                effect_sizes.append(effect_size)
                standard_errors.append(se)
                weights.append(weight)
                
                title = paper_data.get("title", "Unknown")[:30]
                print(f"{title}...: ES={effect_size:.3f}, SE={se:.3f}")
                
                # 提取作者信息
                author = paper_data.get("author") or "Unknown"
                
                # 清理作者姓名（只保留姓氏）
                if author != "Unknown":
                    if "," in author:
                        author = author.split(",")[0].strip()
                    elif " " in author:
                        author = author.split()[-1].strip()
                
                # 创建StudyDetail对象
                study_detail = StudyDetail(
                    paper_id=paper["paper_id"],
                    doi=paper_data.get("doi", paper["paper_id"]),
                    title=paper_data.get("title", "Unknown"),
                    first_author=author,
                    year=paper_data.get("year"),
                    sample_size=outcome_data["intervention_group"]["n"] + outcome_data["control_group"]["n"],
                    intervention_n=outcome_data["intervention_group"]["n"],
                    control_n=outcome_data["control_group"]["n"],
                    effect_size=effect_size,
                    se=se,
                    weight=weight,
                    ci_lower=effect_size - 1.96 * se,
                    ci_upper=effect_size + 1.96 * se
                )
                
                study_details.append(study_detail)
                
            except Exception as e:
                title = paper_data.get("title", "Unknown")[:30]
                print(f"计算效应量失败 ({title}...): {e}")
                # 打印更详细的调试信息
                print(f"论文ID: {paper['paper_id']}")
                print(f"结局指标数据: {outcome_data}")
                continue
        
        if len(effect_sizes) < 2:
            raise ValueError(f"成功计算效应量的研究不足 ({len(effect_sizes)} < 2)")
        
        # 异质性检验
        heterogeneity = self._calculate_heterogeneity(effect_sizes, standard_errors, weights)

        print(f"\n=== 异质性检验结果 ===")
        print(f"Q统计量: {heterogeneity['q']:.3f}")
        print(f"自由度: {heterogeneity['df']}")
        if heterogeneity['p_value'] < 0.001:
            print(f"异质性p值: <0.001")
        else:
            print(f"异质性p值: {heterogeneity['p_value']:.3f}")
        print(f"I²: {heterogeneity['i2']:.1f}%")
        print(f"τ²: {heterogeneity['tau_squared']:.3f}")

        # 基于异质性检验结果选择模型
        if method == "auto":
            if (heterogeneity['i2'] >= i2_threshold or
                    heterogeneity['p_value'] < heterogeneity_p_threshold):
                selected_method = "random"
                print(f"\n=== 模型选择 ===")
                print(f"检测到显著异质性 (I²={heterogeneity['i2']:.1f}%, p={heterogeneity['p_value']:.3f})")
                print(f"自动选择: 随机效应模型")
            else:
                selected_method = "fixed"
                print(f"\n=== 模型选择 ===")
                print(f"异质性不显著 (I²={heterogeneity['i2']:.1f}%, p={heterogeneity['p_value']:.3f})")
                print(f"自动选择: 固定效应模型")
        else:
            selected_method = method
            print(f"\n=== 模型选择 ===")
            print(f"用户指定: {selected_method}效应模型")
            if method == "fixed" and heterogeneity['i2'] >= i2_threshold:
                print(f"警告: 存在显著异质性 (I²={heterogeneity['i2']:.1f}%)，建议使用随机效应模型")
        
        # 合并效应量
        if selected_method == "fixed":
            pooled_result = self._fixed_effects_meta_analysis(effect_sizes, standard_errors, weights)
        else:  # random effects
            pooled_result = self._random_effects_meta_analysis(
                effect_sizes, standard_errors, weights, heterogeneity["tau_squared"]
            )
        
        # 预测区间
        prediction_interval = self._calculate_prediction_interval(
            pooled_result["effect"], pooled_result["se"], heterogeneity["tau_squared"], len(effect_sizes)
        )
        
        return MetaAnalysisResult(
            outcome_name=outcome_name,
            included_studies=len(filtered_papers),
            total_participants=sum(detail.sample_size for detail in study_details),
            pooled_effect_size=pooled_result["effect"],
            confidence_interval=pooled_result["ci"],
            p_value=pooled_result["p_value"],
            heterogeneity_i2=heterogeneity["i2"],
            heterogeneity_q=heterogeneity["q"],
            heterogeneity_p=heterogeneity["p_value"],
            tau_squared=heterogeneity["tau_squared"],
            prediction_interval=prediction_interval,
            study_details=study_details,
            effect_type=effect_type,
            method=method
        )
    
    def _calculate_heterogeneity(self, effect_sizes: List[float], 
                                standard_errors: List[float], 
                                weights: List[float]) -> Dict:
        """计算异质性统计量"""
        
        effect_sizes = np.array(effect_sizes)
        weights = np.array(weights)
        
        # 固定效应合并效应量
        pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # Q统计量
        q = np.sum(weights * (effect_sizes - pooled_effect) ** 2)
        df = len(effect_sizes) - 1
        q_p_value = 1 - stats.chi2.cdf(q, df) if df > 0 else 1.0
        
        # I²统计量
        i2 = max(0, (q - df) / q) * 100 if q > 0 else 0
        
        # τ²估计 (DerSimonian-Laird方法)
        if df > 0:
            c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
            tau_squared = max(0, (q - df) / c) if c > 0 else 0
        else:
            tau_squared = 0
        
        return {
            "q": q,
            "df": df,
            "p_value": q_p_value,
            "i2": i2,
            "tau_squared": tau_squared
        }
    
    def _fixed_effects_meta_analysis(self, effect_sizes: List[float],
                                   standard_errors: List[float],
                                   weights: List[float]) -> Dict:
        """固定效应模型Meta分析"""
        
        effect_sizes = np.array(effect_sizes)
        weights = np.array(weights)
        
        # 合并效应量
        pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # 标准误
        pooled_se = 1 / np.sqrt(np.sum(weights))
        
        # 95%置信区间
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        
        # Z检验
        z_score = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            "effect": pooled_effect,
            "se": pooled_se,
            "ci": (ci_lower, ci_upper),
            "z_score": z_score,
            "p_value": p_value
        }
    
    def _random_effects_meta_analysis(self, effect_sizes: List[float],
                                    standard_errors: List[float],
                                    weights: List[float],
                                    tau_squared: float) -> Dict:
        """随机效应模型Meta分析"""
        
        effect_sizes = np.array(effect_sizes)
        standard_errors = np.array(standard_errors)
        
        # 随机效应权重
        random_weights = 1 / (standard_errors**2 + tau_squared)
        
        # 合并效应量
        pooled_effect = np.sum(random_weights * effect_sizes) / np.sum(random_weights)
        
        # 标准误
        pooled_se = 1 / np.sqrt(np.sum(random_weights))
        
        # 95%置信区间
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        
        # Z检验
        z_score = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            "effect": pooled_effect,
            "se": pooled_se,
            "ci": (ci_lower, ci_upper),
            "z_score": z_score,
            "p_value": p_value
        }
    
    def _calculate_prediction_interval(self, pooled_effect: float, pooled_se: float,
                                     tau_squared: float, k: int) -> Tuple[float, float]:
        """计算预测区间"""
        
        if k < 3:  # 研究数量太少，无法计算预测区间
            return None
        
        # 预测区间标准误
        pi_se = np.sqrt(pooled_se**2 + tau_squared)
        
        # t分布临界值
        df = k - 2
        t_critical = stats.t.ppf(0.975, df)
        
        pi_lower = pooled_effect - t_critical * pi_se
        pi_upper = pooled_effect + t_critical * pi_se
        
        return (pi_lower, pi_upper)
    
    def generate_forest_plot(self, result: MetaAnalysisResult, save_path: Optional[str] = None):
        """生成森林图"""
        
        if not result.study_details:
            print("没有研究详细信息可以绘制森林图")
            return
        
        study_details = result.study_details
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(study_details) * 0.8)))
        
        y_positions = range(len(study_details))
        effect_sizes = [study.effect_size for study in study_details]
        ci_lowers = [study.ci_lower for study in study_details]
        ci_uppers = [study.ci_upper for study in study_details]
        weights = [study.weight for study in study_details]
        
        # 创建研究标签（作者 年份）
        study_labels = []
        for study in study_details:
            if study.year:
                label = f"{study.first_author} {study.year}"
            else:
                label = study.first_author
            study_labels.append(label)
        
        # 计算权重比例用于调整点的大小
        max_weight = max(weights) if weights else 1
        point_sizes = [50 + (w / max_weight) * 100 for w in weights]
        
        # 绘制各研究的效应量和置信区间
        for i, (es, ci_l, ci_u, size) in enumerate(zip(effect_sizes, ci_lowers, ci_uppers, point_sizes)):
            # 绘制置信区间
            ax.plot([ci_l, ci_u], [i, i], 'k-', linewidth=1, alpha=0.7)
            ax.plot([ci_l, ci_l], [i-0.1, i+0.1], 'k-', linewidth=1, alpha=0.7)
            ax.plot([ci_u, ci_u], [i-0.1, i+0.1], 'k-', linewidth=1, alpha=0.7)
            
            # 绘制效应量点（大小反映权重）
            ax.scatter([es], [i], s=size, c='blue', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 绘制合并效应量
        pooled_y = len(study_details) + 0.5
        pooled_ci_lower, pooled_ci_upper = result.confidence_interval
        
        # 合并效应量的置信区间
        ax.plot([pooled_ci_lower, pooled_ci_upper], [pooled_y, pooled_y], 'r-', linewidth=3, alpha=0.8)
        ax.plot([pooled_ci_lower, pooled_ci_lower], [pooled_y-0.15, pooled_y+0.15], 'r-', linewidth=3, alpha=0.8)
        ax.plot([pooled_ci_upper, pooled_ci_upper], [pooled_y-0.15, pooled_y+0.15], 'r-', linewidth=3, alpha=0.8)
        
        # 合并效应量点
        ax.scatter([result.pooled_effect_size], [pooled_y], s=200, c='red', marker='D', 
                  edgecolors='black', linewidth=1, label=f'Pooled Effect: {result.pooled_effect_size:.3f}')
        
        # 绘制无效线
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # 设置标签
        all_labels = study_labels + ['Pooled']
        all_positions = list(y_positions) + [pooled_y]
        
        ax.set_yticks(all_positions)
        ax.set_yticklabels(all_labels)
        ax.set_xlabel(f'Effect Size ({result.effect_type.upper()})')
        ax.set_title(f'Forest Plot: {result.outcome_name}')
        ax.legend()
        
        # 添加统计信息文本框
        stats_text = (f'Studies: {result.included_studies}\n'
                     f'Participants: {result.total_participants}\n'
                     f'Method: {result.method.title()}\n'
                     f'I²: {result.heterogeneity_i2:.1f}%\n'
                     f'p-value: {result.p_value:.3f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"森林图已保存到: {save_path}")
        
        plt.show()
        
        # 打印研究详细信息
        print(f"\n=== 纳入研究详细信息 ===")
        for i, study in enumerate(study_details, 1):
            print(f"{i}. {study.first_author} {study.year}")
            print(f"   DOI: {study.doi}")
            print(f"   样本量: {study.sample_size} (干预组: {study.intervention_n}, 对照组: {study.control_n})")
            print(f"   效应量: {study.effect_size:.3f} [{study.ci_lower:.3f}, {study.ci_upper:.3f}]")
            print()
    
    def print_meta_analysis_summary(self, result: MetaAnalysisResult):
        """打印Meta分析结果摘要"""
        
        print(f"\n=== Meta分析结果: {result.outcome_name} ===")
        print(f"纳入研究数量: {result.included_studies}")
        print(f"总参与者数量: {result.total_participants}")
        print(f"合并效应量: {result.pooled_effect_size:.3f}")
        print(f"95%置信区间: ({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})")
        if result.p_value < 0.001:
            print(f"p值: <0.001")
        else:
            print(f"p值: {result.p_value:.3f}")
        print(f"\n异质性检验:")
        print(f"  I²: {result.heterogeneity_i2:.1f}%")
        print(f"  Q: {result.heterogeneity_q:.2f}")
        print(f"  τ²: {result.tau_squared:.3f}")
        print(f"  异质性p值: {result.heterogeneity_p:.3f}")
        
        if result.prediction_interval:
            print(f"\n95%预测区间: ({result.prediction_interval[0]:.3f}, {result.prediction_interval[1]:.3f})")
        
        # 解释异质性
        if result.heterogeneity_i2 < 25:
            heterogeneity_level = "低"
        elif result.heterogeneity_i2 < 50:
            heterogeneity_level = "中等"
        elif result.heterogeneity_i2 < 75:
            heterogeneity_level = "较高"
        else:
            heterogeneity_level = "高"
        
        print(f"\n异质性水平: {heterogeneity_level}")
        
        # 效应量解释
        abs_effect = abs(result.pooled_effect_size)
        if abs_effect < 0.2:
            effect_magnitude = "微小"
        elif abs_effect < 0.5:
            effect_magnitude = "小"
        elif abs_effect < 0.8:
            effect_magnitude = "中等"
        else:
            effect_magnitude = "大"
        
        print(f"效应量大小: {effect_magnitude}")
        
        if result.p_value < 0.05:
            print("结果具有统计学意义")
        else:
            print("结果无统计学意义")

    async def perform_targeted_meta_analysis(
        self,
        target_paper_ids: List[str],
        outcome_name: str,
        query: str = "",
        effect_type: str = "smd",
        method: str = "auto",
        i2_threshold: float = 50.0,
        heterogeneity_p_threshold: float = 0.05,
        timepoint_selection: str = "auto",
        **filter_kwargs
    ) -> MetaAnalysisResult:
        """对指定论文列表执行Meta分析"""

        print(f"\n=== 对指定论文进行Meta分析: {outcome_name} ===")
        print(f"用户查询: {query}")
        print(f"时间点选择策略: {timepoint_selection}")
        print(f"目标论文数量: {len(target_paper_ids)}")
        
        # 临时保存原始数据
        original_papers_data = self.papers_data
        
        # 筛选出目标论文的数据
        targeted_papers_data = {}
        for paper_id in target_paper_ids:
            if paper_id in self.papers_data:
                targeted_papers_data[paper_id] = self.papers_data[paper_id]
            else:
                print(f"警告: 论文 {paper_id} 不在数据库中")
        
        print(f"实际可用论文数量: {len(targeted_papers_data)}")
        
        if len(targeted_papers_data) < 2:
            raise ValueError(f"可用论文数量不足 ({len(targeted_papers_data)} < 2)")
        
        try:
            # 临时替换papers_data
            self.papers_data = targeted_papers_data
            
            # 调用Meta分析方法（传入query和timepoint_selection）
            result = await self.perform_meta_analysis(
                outcome_name=outcome_name,
                query=query,
                effect_type=effect_type,
                method=method,
                i2_threshold=i2_threshold,
                heterogeneity_p_threshold=heterogeneity_p_threshold,
                timepoint_selection=timepoint_selection,
                **filter_kwargs
            )
            
            return result
            
        finally:
            # 恢复原始数据
            self.papers_data = original_papers_data

    def get_available_outcomes_for_papers(self, target_paper_ids: List[str]) -> Dict[str, List[str]]:
        """获取指定论文中可用的结局指标（合并同义词，避免重复计算）"""
        
        outcomes = {"primary": [], "secondary": []}
        outcome_counts = {}
        
        for paper_id in target_paper_ids:
            if paper_id not in self.papers_data:
                continue
            
            paper_data = self.papers_data[paper_id]
            
            # 用于记录当前论文已处理的标准指标名称，避免重复计算
            processed_outcomes_in_paper = set()
            
            # 主要结局指标
            for outcome in paper_data.get("primary_outcomes", []):
                outcome_name = outcome.get("outcome_name", "")
                if outcome_name:
                    # 跳过baseline数据
                    if self._is_baseline_only(outcome.get("time_point", "")):
                        continue
                    
                    standard_name = self._get_standard_outcome_name(outcome_name)
                
                    # 避免同一论文的同一指标被重复计算
                    if standard_name not in processed_outcomes_in_paper:
                        processed_outcomes_in_paper.add(standard_name)
                        
                        if standard_name not in outcomes["primary"]:
                            outcomes["primary"].append(standard_name)
                        outcome_counts[standard_name] = outcome_counts.get(standard_name, 0) + 1
            
            # 次要结局指标
            for outcome in paper_data.get("secondary_outcomes", []):
                outcome_name = outcome.get("outcome_name", "")
                if outcome_name:
                    # 跳过baseline数据
                    if self._is_baseline_only(outcome.get("time_point", "")):
                        continue
                    
                    standard_name = self._get_standard_outcome_name(outcome_name)
                
                    # 避免同一论文的同一指标被重复计算
                    if standard_name not in processed_outcomes_in_paper:
                        processed_outcomes_in_paper.add(standard_name)
                        
                        if standard_name not in outcomes["secondary"]:
                            outcomes["secondary"].append(standard_name)
                        outcome_counts[standard_name] = outcome_counts.get(standard_name, 0) + 1
        
        print(f"\n=== 指定论文中的结局指标统计（合并同义词后，避免重复） ===")
        viable_outcomes = []
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{outcome}: {count}篇论文")
            if count >= 2:
                viable_outcomes.append(outcome)
        
        print(f"\n可进行Meta分析的结局指标 ({len(viable_outcomes)}个):")
        for outcome in viable_outcomes:
            print(f"- {outcome}")
        
        return outcomes, viable_outcomes

    def _get_standard_outcome_name(self, outcome_name: str) -> str:
        """获取结局指标的标准名称"""
        if not outcome_name:
            return outcome_name

        outcome_clean = re.sub(r'[^\w\s]', '', outcome_name.lower().strip())
        # print(f"    标准化处理: '{outcome_name}' -> '{outcome_clean}'")

        # 在别名配置中查找匹配
        for standard_key, aliases in OUTCOME_ALIASES.items():
            for alias in aliases:
                if (outcome_clean == alias or
                        outcome_clean in alias or
                        alias in outcome_clean):
                    # print(f"    找到匹配: '{outcome_clean}' 匹配到 '{standard_key}' (通过别名 '{alias}')")
                    return standard_key.upper()  # 返回大写的标准名称

        # 如果没找到匹配，返回原名称
        # print(f"    未找到匹配，返回原名称: '{outcome_name}'")
        return outcome_name


def main():
    """主函数：完整的Meta分析流程"""
    
    print("=== TMS论文Meta分析系统 ===")
    
    # 创建GraphRAG实例
    rag = MetaAnalysisGraphRAG(working_dir="D:/YJS/TMSrag/rTMS-rag/nano_graphrag/meta_analysis_cache")
    
    # 创建Meta分析器
    analyzer = MetaAnalyzer(rag)

    loop = always_get_an_event_loop()
    loop.run_until_complete(analyzer.load_papers_data())
    
    # 1. 分析数据完整性
    outcome_counts = analyzer.analyze_data_completeness()
    
    # 2. 找出有足够数据的结局指标
    viable_outcomes = [outcome for outcome, count in outcome_counts.items() if count >= 2]
    
    if not viable_outcomes:
        print("\n没有找到有足够数据进行Meta分析的结局指标")
        return
    
    print(f"\n=== 可进行Meta分析的结局指标 ===")
    for outcome in viable_outcomes:
        print(f"- {outcome} ({outcome_counts[outcome]}篇论文)")
    
    # 3. 对每个可行的结局指标进行Meta分析
    successful_analyses = []
    
    for outcome in viable_outcomes:
        print(f"\n{'='*60}")
        print(f"开始Meta分析: {outcome}")
        print(f"{'='*60}")
        
        try:
            result = loop.run_until_complete(analyzer.perform_meta_analysis(
                outcome_name=outcome,
                effect_type="smd",
                method="auto",
                min_sample_size=5,
                max_bias_risk="High",
                require_complete_data=True
            ))
            
            # 打印结果
            analyzer.print_meta_analysis_summary(result)
            
            # 生成森林图
            plot_filename = f"{outcome.replace(' ', '_').replace('/', '_')}_forest_plot.png"
            analyzer.generate_forest_plot(result, plot_filename)
            
            successful_analyses.append((outcome, result))
            
        except ValueError as e:
            print(f" {outcome} Meta分析失败: {e}")
        except Exception as e:
            print(f" {outcome} Meta分析出错: {e}")
    
    # 4. 总结
    print(f"\n{'='*60}")
    print(f"Meta分析总结")
    print(f"{'='*60}")
    print(f"总论文数: {len(analyzer.papers_data)}")
    print(f"可分析的结局指标: {len(viable_outcomes)}")
    print(f"成功完成的Meta分析: {len(successful_analyses)}")
    
    if successful_analyses:
        print(f"\n成功的Meta分析:")
        for outcome, result in successful_analyses:
            print(f"- {outcome}: ES={result.pooled_effect_size:.3f}, "
                  f"p={result.p_value:.3f}, I²={result.heterogeneity_i2:.1f}%")


if __name__ == "__main__":
    main()