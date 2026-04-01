"""STS2-Agent API client for communicating with the game"""

import json
import time
from typing import Any, Dict, Optional, List

import requests
from requests.exceptions import RequestException


class Sts2Client:
    """Client for interacting with STS2-Agent HTTP API"""

    def __init__(self, host: str = "localhost", port: int = 8080, timeout: float = 10.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_health(self) -> Dict[str, Any]:
        """Get health status of the STS2-Agent"""
        url = f"{self.base_url}/health"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            return data["data"]
        return data

    def get_state(self) -> Dict[str, Any]:
        """Get current game state"""
        try:
            url = f"{self.base_url}/state"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data is not None and isinstance(data, dict):
                if "data" in data and data["data"] is not None and isinstance(data["data"], dict):
                    return data["data"]
                return data
            return {}
        except Exception as e:
            print(f"Error getting game state: {e}")
            return {}


    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available actions"""
        url = f"{self.base_url}/actions/available"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if "data" in data and "actions" in data["data"]:
            return data["data"]["actions"]
        return []

    def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a game action"""
        try:
            url = f"{self.base_url}/action"
            # Build the action payload
            action_payload = {
                "action": action,
                "card_index": kwargs.get("card_index"),
                "target_index": kwargs.get("target_index"),
                "option_index": kwargs.get("option_index"),
                "command": kwargs.get("command"),
                "client_context": kwargs.get("client_context", {
                    "source": "mcp",
                    "tool_name": action,
                }),
            }
            
            # Filter out None values
            action_payload = {k: v for k, v in action_payload.items() if v is not None}
            
            response = self.session.post(url, json=action_payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict):
                    return data["data"]
                return data
            return {}
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return {}


    def end_turn(self) -> Dict[str, Any]:
        """End the current turn"""
        return self.execute_action("end_turn")

    def play_card(self, card_index: int, target_index: int | None = None) -> Dict[str, Any]:
        """Play a card"""
        return self.execute_action("play_card", card_index=card_index, target_index=target_index)

    def continue_run(self) -> Dict[str, Any]:
        """Continue the current run"""
        return self.execute_action("continue_run")

    def abandon_run(self) -> Dict[str, Any]:
        """Abandon the current run"""
        return self.execute_action("abandon_run")

    def open_character_select(self) -> Dict[str, Any]:
        """Open character selection menu"""
        return self.execute_action("open_character_select")

    def open_timeline(self) -> Dict[str, Any]:
        """Open timeline menu"""
        return self.execute_action("open_timeline")

    def close_main_menu_submenu(self) -> Dict[str, Any]:
        """Close main menu submenu"""
        return self.execute_action("close_main_menu_submenu")

    def choose_timeline_epoch(self, option_index: int) -> Dict[str, Any]:
        """Choose timeline epoch"""
        return self.execute_action("choose_timeline_epoch", option_index=option_index)

    def confirm_timeline_overlay(self) -> Dict[str, Any]:
        """Confirm timeline overlay"""
        return self.execute_action("confirm_timeline_overlay")

    def choose_map_node(self, option_index: int) -> Dict[str, Any]:
        """Choose map node"""
        return self.execute_action("choose_map_node", option_index=option_index)

    def collect_rewards_and_proceed(self) -> Dict[str, Any]:
        """Collect rewards and proceed"""
        return self.execute_action("collect_rewards_and_proceed")

    def claim_reward(self, option_index: int) -> Dict[str, Any]:
        """Claim reward"""
        return self.execute_action("claim_reward", option_index=option_index)

    def choose_reward_card(self, option_index: int) -> Dict[str, Any]:
        """Choose reward card"""
        return self.execute_action("choose_reward_card", option_index=option_index)

    def skip_reward_cards(self) -> Dict[str, Any]:
        """Skip reward cards"""
        return self.execute_action("skip_reward_cards")

    def select_deck_card(self, option_index: int) -> Dict[str, Any]:
        """Select deck card"""
        return self.execute_action("select_deck_card", option_index=option_index)

    def confirm_selection(self) -> Dict[str, Any]:
        """Confirm selection"""
        return self.execute_action("confirm_selection")

    def proceed(self) -> Dict[str, Any]:
        """Proceed"""
        return self.execute_action("proceed")

    def open_chest(self) -> Dict[str, Any]:
        """Open chest"""
        return self.execute_action("open_chest")

    def choose_treasure_relic(self, option_index: int) -> Dict[str, Any]:
        """Choose treasure relic"""
        return self.execute_action("choose_treasure_relic", option_index=option_index)

    def choose_event_option(self, option_index: int) -> Dict[str, Any]:
        """Choose event option"""
        return self.execute_action("choose_event_option", option_index=option_index)

    def choose_rest_option(self, option_index: int) -> Dict[str, Any]:
        """Choose rest option"""
        return self.execute_action("choose_rest_option", option_index=option_index)

    def open_shop_inventory(self) -> Dict[str, Any]:
        """Open shop inventory"""
        return self.execute_action("open_shop_inventory")

    def close_shop_inventory(self) -> Dict[str, Any]:
        """Close shop inventory"""
        return self.execute_action("close_shop_inventory")

    def buy_card(self, option_index: int) -> Dict[str, Any]:
        """Buy card"""
        return self.execute_action("buy_card", option_index=option_index)

    def buy_relic(self, option_index: int) -> Dict[str, Any]:
        """Buy relic"""
        return self.execute_action("buy_relic", option_index=option_index)

    def buy_potion(self, option_index: int) -> Dict[str, Any]:
        """Buy potion"""
        return self.execute_action("buy_potion", option_index=option_index)

    def remove_card_at_shop(self) -> Dict[str, Any]:
        """Remove card at shop"""
        return self.execute_action("remove_card_at_shop")

    def select_character(self, option_index: int) -> Dict[str, Any]:
        """Select character"""
        return self.execute_action("select_character", option_index=option_index)

    def embark(self) -> Dict[str, Any]:
        """Embark on a new run"""
        return self.execute_action("embark")

    def unready(self) -> Dict[str, Any]:
        """Unready"""
        return self.execute_action("unready")

    def increase_ascension(self) -> Dict[str, Any]:
        """Increase ascension level"""
        return self.execute_action("increase_ascension")

    def decrease_ascension(self) -> Dict[str, Any]:
        """Decrease ascension level"""
        return self.execute_action("decrease_ascension")

    def use_potion(self, option_index: int, target_index: int | None = None) -> Dict[str, Any]:
        """Use potion"""
        return self.execute_action("use_potion", option_index=option_index, target_index=target_index)

    def discard_potion(self, option_index: int) -> Dict[str, Any]:
        """Discard potion"""
        return self.execute_action("discard_potion", option_index=option_index)

    def run_console_command(self, command: str) -> Dict[str, Any]:
        """Run console command"""
        return self.execute_action("run_console_command", command=command)

    def confirm_modal(self) -> Dict[str, Any]:
        """Confirm modal"""
        return self.execute_action("confirm_modal")

    def dismiss_modal(self) -> Dict[str, Any]:
        """Dismiss modal"""
        return self.execute_action("dismiss_modal")

    def return_to_main_menu(self) -> Dict[str, Any]:
        """Return to main menu"""
        return self.execute_action("return_to_main_menu")

    def is_connected(self) -> bool:
        """Check if connection to game is active"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_connection(self, max_retries: int = 30, retry_interval: float = 2.0) -> bool:
        """Wait for game to be available"""
        for i in range(max_retries):
            if self.is_connected():
                return True
            if i < max_retries - 1:
                time.sleep(retry_interval)
        return False

    def close(self):
        """Close the session"""
        self.session.close()