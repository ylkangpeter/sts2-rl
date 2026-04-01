# -*- coding: utf-8 -*-
"""Environment factory for creating headless or UI environments"""

from typing import Dict, Any


def create_environment(mode: str, **kwargs):
    """
    创建环境

    Args:
        mode: "headless" 或 "ui"
        **kwargs: 环境参数

    Returns:
        UnifiedSTS2Env 实例
    """
    # 延迟导入避免循环依赖
    if mode in ("headless", "http_cli_rl"):
        from .http_cli_rl import HttpCliRlEnv
        return HttpCliRlEnv(**kwargs)
    elif mode == "legacy_headless":
        from .headless import HeadlessSTS2Env
        return HeadlessSTS2Env(**kwargs)
    elif mode == "ui":
        from .ui import UISTS2Env
        return UISTS2Env(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'headless', 'http_cli_rl', 'legacy_headless', or 'ui'.")


def create_from_config(config: Dict[str, Any]):
    """
    从配置创建环境

    Args:
        config: 配置字典

    Returns:
        UnifiedSTS2Env 实例
    """
    mode = config.get('mode', 'headless')
    env_config = config.get('environment', {})
    return create_environment(mode, **env_config)


class EnvironmentFactory:
    """环境工厂（向后兼容）"""

    @staticmethod
    def create(mode: str, **kwargs):
        return create_environment(mode, **kwargs)

    @staticmethod
    def create_from_config(config: Dict[str, Any]):
        return create_from_config(config)
