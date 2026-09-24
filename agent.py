import json
import os
from openai import OpenAI

class WorldCupAgent:
    def __init__(self):
        # 【关键修复】清空代理环境变量，强制 Python 绕过翻墙软件直连本地
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['ALL_PROXY'] = ''
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
        
        # 将 localhost 改为 127.0.0.1，避免 Windows IPv6 解析干扰
        self.client = OpenAI(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            timeout=120.0 
        )
        self.model = "qwen2.5:7b"
        print(f"[System] : {self.model}")

    # === 新增：大模型有限记忆窗口 (Context Window) ===
        # 强制截断：模拟注意力的局限性，Agent 只能记住最近发生的 2 个事件
        self.short_term_memory = []
        self.memory_limit = 2  
        
        print(f"[System] 代理已绕过，直连本地 Ollama 大脑: {self.model}")

    def get_action_and_belief(self, observation, target_team):
        # 1. 记忆更新与截断 (强制诱发近因效应)
        current_time = observation['match_event']['time']
        current_event = observation['match_event']['event']
        self.short_term_memory.append(f"[{current_time}'] {current_event}")
        
        # 超过记忆限制时，遗忘早期的宏观事件
        if len(self.short_term_memory) > self.memory_limit:
            self.short_term_memory.pop(0)
            
        memory_context = "\n".join(self.short_term_memory)
        
        prompt = f"""
        You are a rational sports betting trading algorithm. 
        
        Recent Match History (Strictly limited memory):
        {memory_context}
        
        Current Market Depth (LOB): {observation['market_depth']}
        Your Portfolio: {observation['portfolio']}
        
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
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        raw_content = response.choices[0].message.content.strip().replace('```json', '').replace('```', '')
        
        try:
            action_data = json.loads(raw_content)
            b_t = action_data.get("subjective_win_prob", 0.5) 
        except Exception:
            action_data = {"action": "wait", "volume": 0, "thought": "parse error"}
            b_t = 0.5
            
        logprobs = "Local Ollama response captured"
            
        return action_data, b_t, logprobs