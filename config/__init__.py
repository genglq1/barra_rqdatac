# -*- coding: utf-8 -*-
"""
配置加载层
==========
集中读取 config/ 下的 yaml 文件,供 data/factors/model/analysis 各层调用。
用法:
    from config import get_settings, get_weights

    settings = get_settings()
    weights = get_weights("cne5")

设计:懒加载 + 缓存,避免重复读盘。若未安装 pyyaml,回退到内置 config.py。
"""

import os

# 项目根目录(barra_rqdatac/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

_settings_cache = None
_weights_cache = {}


def _load_yaml(filepath):
    """读取 yaml 文件为 dict。优先用 pyyaml,缺失则给出明确提示。"""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "缺少 pyyaml 依赖,请先安装: pip install pyyaml\n"
            "或改用 config/config.py(纯Python)替代 yaml。"
        ) from e

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"配置文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_settings():
    """
    读取主配置 settings.yaml,返回 dict。
    并将相对路径解析为基于 PROJECT_ROOT 的绝对路径,方便各层直接使用。

    返回结构(关键字段):
        settings["rqdatac"]["license"]
        settings["paths"]["base"]  -> 绝对路径
        settings["paths"]["factor_dir"](version)  -> 拼接好的因子产出目录
        settings["market"]["benchmark"]
        ...
    """
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    settings = _load_yaml(os.path.join(CONFIG_DIR, "settings.yaml"))

    # 把 paths 下的相对路径转绝对路径
    paths = settings.get("paths", {})
    for key, val in paths.items():
        if isinstance(val, str):
            paths[key] = os.path.join(PROJECT_ROOT, val.replace("/", os.sep))
    settings["paths"] = paths

    _settings_cache = settings
    return settings


def get_weights(version="cne5"):
    """
    读取指定版本的因子合成权重。

    参数:
        version: "cne5" 或 "cne6"
    返回:
        dict: {风格因子名: {描述因子名: 权重}}
    """
    if version in _weights_cache:
        return _weights_cache[version]

    filepath = os.path.join(CONFIG_DIR, f"{version}_weights.yaml")
    data = _load_yaml(filepath)
    weights = data.get("style_factors", {}) if data else {}
    _weights_cache[version] = weights
    return weights


def get_path(*keys):
    """
    快捷取路径。例如 get_path("base") 返回基础数据目录绝对路径。
    """
    settings = get_settings()
    node = settings["paths"]
    for k in keys:
        node = node[k]
    return node


def get_factor_dir(version="cne5"):
    """因子产出目录: data_store/factors/{version}/"""
    settings = get_settings()
    return os.path.join(settings["paths"]["factors"], version)


def get_model_dir(version="cne5"):
    """模型产出目录: data_store/model/{version}/"""
    settings = get_settings()
    return os.path.join(settings["paths"]["model"], version)


def get_batch_size():
    """rqdatac 单次请求股票数(来自 settings.yaml rqdatac.batch_size)。"""
    return get_settings()["rqdatac"].get("batch_size", 800)


def get_start_date():
    """历史数据默认起始日(来自 settings.yaml date_range.start_date)。"""
    return get_settings().get("date_range", {}).get("start_date", "2010-01-01")


def get_factor_return_start_date():
    """因子收益率回归默认起始日(date_range.factor_return_start_date,缺省 2022-01-01)。"""
    return get_settings().get("date_range", {}).get(
        "factor_return_start_date", "2022-01-01")
