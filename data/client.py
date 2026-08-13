# -*- coding: utf-8 -*-
"""
rqdatac 连接封装
================
1. init_rqdatac(): 从 config 读取 license 并初始化 rqdatac(单例,全局只初始化一次)。
2. 代码后缀转换:Tushare/Wind 风格(.SH/.SZ/.BJ) <-> rqdatac 风格(.XSHG/.XSHE/.XBJG)。

注意:rqdatac 的 order_book_id 一律带交易所后缀:
    上交所 .XSHG / 深交所 .XSHE / 北交所 .XBJG
而原项目(及多数国内习惯)用 .SH / .SZ / .BJ。本模块统一在数据进出时做转换。
"""

import os
import re

# rqdatac 是可选依赖,未安装时本模块仍可 import(仅在实际调用 API 时报错)
try:
    import rqdatac
    _RQDATAC_AVAILABLE = True
except ImportError:
    rqdatac = None
    _RQDATAC_AVAILABLE = False

from config import get_settings

_initialized = False


def init_rqdatac(force=False):
    """
    初始化 rqdatac 连接(单例)。

    参数:
        force: True 时强制重新初始化
    返回:
        rqdatac 模块本身
    异常:
        - 未安装 rqdatac: ImportError
        - license 为空: ValueError
        - license 无效: rqdatac 内部异常
    """
    global _initialized
    if _initialized and not force:
        return rqdatac

    if not _RQDATAC_AVAILABLE:
        raise ImportError(
            "未安装 rqdatac,请先安装: pip install rqdatac"
        )

    settings = get_settings()
    rq_cfg = settings.get("rqdatac", {})
    license_key = rq_cfg.get("license", "").strip()

    if not license_key:
        raise ValueError(
            "rqdatac license 未配置。请在 config/settings.yaml 的 "
            "rqdatac.license 字段填入你的 license key(米筐邮件中获取)。"
        )

    # 初始化(支持 license 字符串直接传入)
    rqdatac.init(license=license_key)
    _initialized = True
    return rqdatac


# ----------------------------------------------------------------------------
# 代码后缀转换工具
# ----------------------------------------------------------------------------

# Wind/Tushare 后缀 -> rqdatac 后缀
_SUFFIX_TO_RQ = {
    ".SH": ".XSHG",
    ".SZ": ".XSHE",
    ".BJ": ".XBJG",
}
# 反向映射
_SUFFIX_FROM_RQ = {v: k for k, v in _SUFFIX_TO_RQ.items()}


def to_rqcode(code):
    """
    将 Wind/Tushare 风格代码转为 rqdatac 风格。
    例: '600519.SH' -> '600519.XSHG'; '000001.SZ' -> '000001.XSHE'
        已是 rqdatac 风格的代码原样返回。
    """
    if not isinstance(code, str):
        return code
    for wsuffix, rsuffix in _SUFFIX_TO_RQ.items():
        if code.endswith(wsuffix):
            return code[: -len(wsuffix)] + rsuffix
    return code  # 已是 rq 风格或无后缀,原样返回


def to_windcode(code):
    """
    将 rqdatac 风格代码转为 Wind/Tushare 风格(便于与原项目数据对齐)。
    例: '600519.XSHG' -> '600519.SH'
    """
    if not isinstance(code, str):
        return code
    for rsuffix, wsuffix in _SUFFIX_FROM_RQ.items():
        if code.endswith(rsuffix):
            return code[: -len(rsuffix)] + wsuffix
    return code


def to_rqcodes(codes):
    """批量转换代码列表到 rqdatac 风格。"""
    return [to_rqcode(c) for c in codes]


def to_windcodes(codes):
    """批量转换代码列表到 Wind 风格。"""
    return [to_windcode(c) for c in codes]
