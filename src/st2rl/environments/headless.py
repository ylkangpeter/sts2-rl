# -*- coding: utf-8 -*-
"""Headless environment using sts2-cli simulator"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

from .base import UnifiedSTS2Env


class HeadlessSTS2Env(UnifiedSTS2Env):
    """使用 headless 模拟器的 STS2 环境"""

    def __init__(
        self,
        dotnet_path: str = None,
        project_path: str = None,
        character: str = "Ironclad",
        max_steps: int = 500,
        seed: str = None,
        verbose: bool = False,
        **kwargs
    ):
        # 先设置 headless 特有参数
        self.dotnet_path = dotnet_path or self._find_dotnet()
        self.project_path = project_path or self._find_project()
        self.seed = seed
        self.verbose = verbose
        self.proc = None

        # 调用父类初始化
        super().__init__(
            mode="headless",
            character=character,
            max_steps=max_steps,
            **kwargs
        )

    def _find_dotnet(self) -> str:
        """查找 dotnet 可执行文件"""
        candidates = [
            r"C:\Program Files\dotnet\dotnet.exe",
            r"C:\Program Files (x86)\dotnet\dotnet.exe",
            "dotnet",
        ]
        for path in candidates:
            if os.path.isfile(path) or path == "dotnet":
                return path
        return "dotnet"

    def _find_project(self) -> str:
        """查找 Sts2Headless 项目"""
        env_root = os.environ.get("STS2_CLI_ROOT")
        candidates = []
        if env_root:
            candidates.append(Path(env_root) / "src" / "Sts2Headless" / "Sts2Headless.csproj")
        candidates.append(Path(__file__).resolve().parents[4] / "sts2-cli" / "src" / "Sts2Headless" / "Sts2Headless.csproj")
        candidates.append(Path(__file__).resolve().parents[4] / "my-sts2-cli" / "src" / "Sts2Headless" / "Sts2Headless.csproj")

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        return str(candidates[0])

    def _init_client(self, **kwargs):
        """初始化客户端（headless 不需要持久连接）"""
        pass

    def _start_simulator(self):
        """启动模拟器"""
        if self.proc:
            self._cleanup()

        # 设置环境变量
        env = os.environ.copy()
        sts2_game_dir = env.get("STS2_GAME_DIR")
        if not sts2_game_dir:
            # 尝试自动查找
            candidates = [
                r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2",
            ]
            for path in candidates:
                if os.path.isdir(path):
                    env["STS2_GAME_DIR"] = path
                    break

        # 检查游戏目录结构
        if sts2_game_dir and os.path.exists(os.path.join(sts2_game_dir, "data_sts2_windows_x86_64")):
            # Godot 版本，需要设置正确的路径
            data_dir = os.path.join(sts2_game_dir, "data_sts2_windows_x86_64")
            env["PATH"] = data_dir + ";" + env.get("PATH", "")
            # 对于 Godot 版本，需要设置 STS2_GAME_DIR 为 data 目录
            env["STS2_GAME_DIR"] = data_dir
            print(f"Set STS2_GAME_DIR to data directory: {data_dir}")

        # 启动进程
        self.proc = subprocess.Popen(
            [self.dotnet_path, "run", "--no-build", "--project", self.project_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if not self.verbose else None,
            text=False,  # 使用二进制模式
            bufsize=0,
            env=env
        )

        # 等待就绪
        ready = self._read_json()
        if ready.get("type") != "ready":
            raise RuntimeError(f"Simulator not ready: {ready}")

        if self.verbose:
            print(f"Headless simulator ready: {ready}")

    def _cleanup(self):
        """清理进程"""
        if self.proc:
            try:
                # 发送退出命令
                self._send({"cmd": "quit"})
                self.proc.wait(timeout=5)
            except:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
                except:
                    self.proc.kill()
            finally:
                self.proc = None

    def _read_json(self) -> dict:
        """读取 JSON 响应"""
        while True:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    return {"type": "error", "message": "EOF - game process ended"}
                
                # 解码为字符串
                try:
                    line_str = line.decode('utf-8').strip()
                except:
                    line_str = line.decode('gbk', errors='ignore').strip()
                
                if not line_str:
                    continue
                    
                if line_str.startswith("{"):
                    return json.loads(line_str)
                # 跳过非 JSON 行
                if self.verbose:
                    print(f"[skip] {line_str[:120]}")
            except json.JSONDecodeError:
                continue
            except Exception as e:
                return {"type": "error", "message": str(e)}

    def _send(self, cmd: dict) -> dict:
        """发送命令"""
        line = json.dumps(cmd)
        if self.verbose:
            print(f"> {line[:200]}")

        try:
            self.proc.stdin.write((line + "\n").encode('utf-8'))
            self.proc.stdin.flush()
            resp = self._read_json()
            if self.verbose:
                print(f"< {json.dumps(resp)[:200]}")
            return resp
        except Exception as e:
            return {"type": "error", "message": str(e)}

    def _do_reset(self):
        """重置环境（启动新游戏）"""
        # 启动模拟器
        self._start_simulator()

        # 生成种子
        if self.seed is None:
            import random
            self.seed = f"headless_{random.randint(0, 100000)}"

        # 开始游戏
        state = self._send({
            "cmd": "start_run",
            "character": self.character,
            "seed": self.seed
        })

        if state.get("type") == "error":
            raise RuntimeError(f"Failed to start run: {state.get('message')}")

        # 保存初始状态
        self._last_raw_state = state

    def _get_raw_state(self) -> Dict[str, Any]:
        """获取原始状态"""
        return getattr(self, '_last_raw_state', {})

    def _execute_action(self, action_type: str, params: Dict[str, Any]):
        """执行动作"""
        # 构建命令
        cmd = {"cmd": "action", "action": action_type, "args": {}}

        # 添加参数
        if action_type == "play_card":
            cmd["args"] = {"card_index": params.get('index', 0)}
            if params.get('target') is not None:
                cmd["args"]["target_index"] = params['target']

        elif action_type == "use_potion":
            cmd["args"] = {"potion_index": params.get('index', 0)}

        elif action_type == "select_map_node":
            cmd["args"] = {
                "col": params.get('index', 0),
                "row": params.get('target', 0)
            }

        elif action_type in ["select_card_reward", "choose_option", "shop_purchase"]:
            cmd["args"] = {"index": params.get('index', 0)}

        elif action_type == "select_bundle":
            cmd["args"] = {"bundle_index": params.get('index', 0)}

        elif action_type == "select_cards":
            cmd["args"] = {"indices": str(params.get('index', 0))}

        # 发送命令
        state = self._send(cmd)

        # 处理错误
        if state.get("type") == "error":
            # 尝试 proceed
            state = self._send({"cmd": "action", "action": "proceed"})

        # 保存状态
        self._last_raw_state = state

    def close(self):
        """关闭环境"""
        self._cleanup()
