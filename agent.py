import json
import os
from openai import OpenAI

class WorldCupAgent:
    def __init__(self):
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("请先设置 MINIMAX_API_KEY 环境变量！")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.minimax.chat/v1"
        )
        self.model = "abab6.5s-chat"
        print(f"[System] 已成功切换至 MiniMax 大脑: {self.model}")

    def get_action_and_belief(self, observation, target_team):
        prompt = f"""
        You are a rational sports betting trading algorithm. 
        Current Match Observation:
        - Match Event: {observation['match_event']}
        - Current Market Depth (LOB): {observation['market_depth']}
        - Your Portfolio: {observation['portfolio']}
        
        Target Team to evaluate: {target_team}
        
        Analyze the situation and decide whether to buy more shares of {target_team} or wait.
        You must respond ONLY with a valid JSON object. Do not use markdown blocks like ```json.
        Exact format required:
        {{
            "thought": "your brief reasoning about the match event and odds",
            "subjective_win_prob": 0.45, 
            "action": "execute_market_order", 
            "team": "{target_team}", 
            "volume": 1500
        }}
        (If you choose to wait, set action to "wait" and volume to 0).
        """
        
        # 移除了触发报错的 response_format 参数
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        # 手动清理大模型可能带有的 Markdown 代码块标记
        raw_content = response.choices[0].message.content.strip().replace('```json', '').replace('```', '')
        
        try:
            action_data = json.loads(raw_content)
            b_t = action_data.get("subjective_win_prob", 0.5) 
        except Exception:
            action_data = {"action": "wait", "volume": 0, "thought": "parse error"}
            b_t = 0.5
            
        logprobs = "MiniMax response captured"
            
        return action_data, b_t, logprobs