# -*- coding: utf-8 -*-
"""
CNE-6 因子注册表(占位)
======================
未来实现 CNE-6 时,在此注册因子。结构与 cne5/registry.py 一致:

    REGISTRY = {
        "size": {
            "func": <计算函数>,
            "outputs": ["lncap", ...],
            "depends_on": [...],
            "base_inputs": [...],
        },
        ...
    }
"""

VERSION = "cne6"

# 占位空注册表;calculate_all 会因空表直接返回
REGISTRY = {}

ALL_DESC_FACTORS = []
ALL_STYLE_FACTORS = []


def get_execution_order():
    return []


def calculate_all(start_date="2010-01-01", end_date=None):
    print("CNE-6 因子集尚未实现(预留骨架)。请先在 factors/cne6/ 实现因子并注册。")
