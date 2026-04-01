# -*- coding: utf-8 -*-
"""UI environment using STS2-Agent and STS2MCP"""

import time
import requests
from typing import Dict, Any, Optional

from .base import UnifiedSTS2Env


class UISTS2Env(UnifiedSTS2Env):
    """使用真实游戏 UI 的 STS2 环境"""

    def __init__(
        self,
        host: str = "localhost",
        agent_port: int = 8080,
        mcp_port: int = 15526,
        character_index: int = 0,
        max_steps: int = 5000,
        use_mcp: bool = True,
        **kwargs
    ):
        # UI 特有参数
        self.host = host
        self.agent_port = agent_port
        self.mcp_port = mcp_port
        self.character_index = character_index
        self.use_mcp = use_mcp

        self.agent_url = f"http://{host}:{agent_port}"
        self.mcp_url = f"http://{host}:{mcp_port}/api/v1/singleplayer"

        # 客户端
        self.agent_client = None
        self.mcp_client = None

        # 调用父类初始化
        super().__init__(
            mode="ui",
            character="Ironclad",  # 通过 character_index 选择
            max_steps=max_steps,
            **kwargs
        )

    def _init_client(self, **kwargs):
        """初始化客户端"""
        # 延迟初始化，在 reset 时连接
        pass

    def _ensure_connection(self):
        """确保连接"""
        # 检查 Agent 连接
        try:
            response = requests.get(f"{self.agent_url}/state", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("STS2-Agent not responding")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to STS2-Agent: {e}")

    def _get_agent_state(self) -> Dict[str, Any]:
        """从 Agent 获取状态"""
        try:
            response = requests.get(f"{self.agent_url}/state", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting agent state: {e}")

        return {"screen": "UNKNOWN"}

    def _get_mcp_state(self) -> Optional[Dict[str, Any]]:
        """从 MCP 获取状态"""
        if not self.use_mcp:
            return None

        try:
            response = requests.get(self.mcp_url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting MCP state: {e}")

        return None

    def _get_raw_state(self) -> Dict[str, Any]:
        """获取原始状态"""
        # 优先从 Agent 获取
        agent_state = self._get_agent_state()

        # 如果 Agent 返回 UNKNOWN，尝试 MCP
        if agent_state.get("screen") == "UNKNOWN" and self.use_mcp:
            mcp_state = self._get_mcp_state()
            if mcp_state:
                # 合并 MCP 状态
                agent_state["mcp_data"] = mcp_state
                # 尝试从 MCP 推断屏幕
                if "state_type" in mcp_state:
                    agent_state["screen"] = mcp_state["state_type"].upper()

        return agent_state

    def _do_reset(self):
        """重置环境（返回主菜单并开始新游戏）"""
        self._ensure_connection()

        # 尝试返回主菜单
        self._return_to_main_menu()

        # 等待一下
        time.sleep(1)

        # 开始新游戏
        self._start_new_game()

        # 等待游戏开始
        time.sleep(2)

    def _return_to_main_menu(self):
        """返回主菜单"""
        # 尝试多种方法
        methods = [
            {"endpoint": "/action", "data": {"action": "return_to_main_menu"}},
            {"endpoint": "/action", "data": {"action": "abandon_run"}},
            {"endpoint": "/action", "data": {"action": "proceed"}},
        ]

        for method in methods:
            try:
                response = requests.post(
                    f"{self.agent_url}{method['endpoint']}",
                    json=method['data'],
                    timeout=5
                )
                time.sleep(0.5)

                # 检查是否回到主菜单
                state = self._get_agent_state()
                if state.get("screen") == "MAIN_MENU":
                    return True
            except Exception as e:
                print(f"Error returning to main menu: {e}")

        return False

    def _start_new_game(self):
        """开始新游戏"""
        # 打开角色选择
        try:
            requests.post(
                f"{self.agent_url}/action",
                json={"action": "open_character_select"},
                timeout=5
            )
            time.sleep(1)

            # 选择角色
            requests.post(
                f"{self.agent_url}/action",
                json={"action": "select_character", "index": self.character_index},
                timeout=5
            )
            time.sleep(1)

            # 开始游戏
            requests.post(
                f"{self.agent_url}/action",
                json={"action": "embark"},
                timeout=5
            )
        except Exception as e:
            print(f"Error starting new game: {e}")

    def _execute_action(self, action_type: str, params: Dict[str, Any]):
        """执行动作"""
        # 构建命令
        action_data = {"action": action_type}

        # 添加参数
        if action_type == "play_card":
            action_data["card_index"] = params.get('index', 0)
            if params.get('target') is not None:
                action_data["target_index"] = params['target']

        elif action_type == "use_potion":
            action_data["slot"] = params.get('index', 0)

        elif action_type == "select_map_node":
            action_data["col"] = params.get('index', 0)
            action_data["row"] = params.get('target', 0)

        elif action_type in ["select_card_reward", "choose_option", "shop_purchase"]:
            action_data["index"] = params.get('index', 0)

        elif action_type == "select_bundle":
            action_data["bundle_index"] = params.get('index', 0)

        elif action_type == "select_cards":
            action_data["indices"] = [params.get('index', 0)]

        # 发送命令到 Agent
        try:
            response = requests.post(
                f"{self.agent_url}/action",
                json=action_data,
                timeout=5
            )

            if response.status_code != 200:
                print(f"Action failed: {response.status_code}")

        except Exception as e:
            print(f"Error executing action: {e}")

        # 等待游戏响应
        time.sleep(0.5)

    def close(self):
        """关闭环境"""
        # 返回主菜单
        self._return_to_main_menu()
