import json
import numpy as np

class WorldCupEnv:
    def __init__(self, json_path="worldcup.json-master/2026/worldcup-full.json", target_team1="Mexico", target_team2="England"):
        self.portfolio = {"balance": 10000, "positions": {"A": 0, "B": 0}}
        self.current_step = 0
        
        # 1. 直接读取传入的完整赛事 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. 定位特定比赛
        match_data = next(m for m in data['matches'] if m['team1'] == target_team1 and m['team2'] == target_team2)
        
        # 3. 自动整合：将物理事件转化为带有赔率的金融时间线
        self.timeline = self._build_financial_timeline(match_data)
        
    def _build_financial_timeline(self, match_data):
        team1 = match_data['team1']
        team2 = match_data['team2']
        events = []
        
        # 提取进球事件 (Goals)
        for g in match_data.get('goals1', []):
            minute = int(g['minute'].split('+')[0]) # 处理 90+6 这种加时
            events.append({"time": minute, "event": f"GOAL! {team1} scores by {g['name']}", "odds_impact": -0.6})
        for g in match_data.get('goals2', []):
            minute = int(g['minute'].split('+')[0])
            events.append({"time": minute, "event": f"GOAL! {team2} scores by {g['name']}", "odds_impact": 0.6})
            
        # 提取红牌事件 (Red Cards)
        for i, team_bookings in enumerate(match_data.get('bookings', [[], []])):
            team_name = team1 if i == 0 else team2
            impact = 0.4 if i == 0 else -0.4 # A队红牌导致A队赔率上升(胜率下降)
            for b in team_bookings:
                if b['type'] == 'R':
                    minute = int(b['minute'].split('+')[0])
                    events.append({"time": minute, "event": f"RED CARD! {team_name} player {b['name']} sent off", "odds_impact": impact})
                    
        # 按时间排序事件
        events.sort(key=lambda x: x['time'])
        
        # 生成时间线与动态赔率
        timeline = [{"time": 0, "event": f"Match Start: {team1} vs {team2}", "true_odds_A": 2.0}]
        current_odds = 2.0
        
        for e in events:
            current_odds += e['odds_impact']
            current_odds = max(1.05, round(current_odds, 2)) # 设定赔率底线防止负数
            timeline.append({
                "time": e['time'], 
                "event": e['event'], 
                "true_odds_A": current_odds
            })
            
        timeline.append({"time": 90, "event": "Match End", "true_odds_A": current_odds})
        return timeline

    def generate_lob(self, mid_price, base_volume=1000):
        """生成呈指数增厚的限价订单薄 (L1-L3)"""
        lob = {}
        for i in range(1, 4):  
            spread = (i - 1) * 0.2  
            price = max(1.01, mid_price - spread) 
            volume = base_volume * (2 ** (i - 1)) 
            lob[f"L{i}"] = {"price": round(price, 2), "volume": volume}
        return lob

    def get_observation(self):
        current_state = self.timeline[self.current_step]
        return {
            "match_event": current_state,
            "market_depth": self.generate_lob(current_state["true_odds_A"]),
            "portfolio": self.portfolio
        }
        
    def step(self, action_dict):
        action = action_dict.get("action")
        reward = 0.0
        
        # 1. 提取当前真实状态 (使用 self.timeline 而不是报错的 self.match_data)
        current_state = self.timeline[self.current_step]
        current_event_text = current_state.get("event", "")
        # 这里假设 true_odds_A 是原本在字典里的键，根据你的 json 结构调整
        true_odds = current_state.get("true_odds_A", 2.0) 
        
        if action == "execute_market_order":
            team = action_dict.get("team", "A") 
            volume = action_dict.get("volume", 0)
            
            if volume > 0:
                # === 引入 L4 级别动态滑点机制 (Market Friction) ===
                # 基础惩罚因子
                base_liquidity_penalty = 0.00005 
                
                # 市场恐慌机制 (Exogenous Shock)：遇到红牌，流动性瞬间枯竭，滑点乘以 4 倍
                panic_multiplier = 4.0 if "RED CARD" in current_event_text else 1.0
                
                # 滑点公式：非线性指数放大机制 (单次买的越多，盘口吃得越深)
                slippage = (volume ** 1.3) * base_liquidity_penalty * panic_multiplier
                
                # 计算吃单后的“有效赔率” (最惨跌至 1.01)
                effective_odds = max(1.01, true_odds - slippage) 
                
                # 计算非理性大额下注带来的真实摩擦成本
                friction_cost = volume * (true_odds - effective_odds)

                # 结算逻辑
                self.portfolio["balance"] -= volume
                self.portfolio["positions"][team] += volume * effective_odds
                
                # 将巨额摩擦成本作为惩罚返回，测试 Agent 会不会因此产生“沉没成本谬误”
                reward = -friction_cost
                
                if friction_cost > 0:
                    print(f"-> ⚠️ Slippage Triggered! Friction Cost: -{friction_cost:.2f}")
        
        # 步进时间
        self.current_step += 1
        
        # 判断比赛是否结束 (根据 timeline 的长度)
        done = self.current_step >= len(self.timeline)
        
        return self.get_observation() if not done else None, reward, done
# 测试整合后的环境
if __name__ == "__main__":
    # 确保 worldcup-full.json 在同一目录下
    env = WorldCupEnv()
    
    print("=== Match Timeline Auto-Generated from worldcup-full.json ===")
    for event in env.timeline:
        print(f"Time: {event['time']}' | Odds (Mexico): {event['true_odds_A']:.2f} | Event: {event['event']}")