import os
from env import WorldCupEnv
from agent import WorldCupAgent

os.environ["API_KEY"] = "sk-xxxxxxxxxxxx"

def run_benchmark():
    # ... 后面的代码保持完全不变
    
    env = WorldCupEnv(target_team1="Mexico", target_team2="England")
    agent = WorldCupAgent()
    
    obs = env.get_observation()
    done = False
    
    print("=== Starting WorldCupBench MVP ===")
    print(f"Initial Portfolio: {obs['portfolio']}")
    
    while not done:
        current_time = obs['match_event']['time']
        current_event = obs['match_event']['event']
        true_odds = obs['match_event']['true_odds_A']
        
        print(f"\n[Time {current_time}'] Event: {current_event}")
        print(f"-> True Market Odds for Mexico: {true_odds:.2f}")
        
        # 1. 代理推理与认知探针截取
        action_dict, b_t, logprobs = agent.get_action_and_belief(obs, target_team="Mexico")
        
        # 2. 计算客观胜率 (p_rational) 和预期校准误差 (ECE)
        p_rational = 1.0 / true_odds
        ece = abs(p_rational - b_t)
        
        print(f"-> Agent Thought: {action_dict.get('thought')}")
        print(f"-> Agent Action : {action_dict.get('action')} | Volume: {action_dict.get('volume')}")
        print(f"-> Cognitive Probe: Objective Prob={p_rational:.2f}, Agent Belief={b_t:.2f} | ECE = {ece:.3f}")

        # 拦截大模型的输出，将真实队名翻译为账本代号
        target = action_dict.get("team")
        if target == "Mexico":
            action_dict["team"] = "A"
        elif target == "England":
            action_dict["team"] = "B"
            
        # 3. 环境步进与滑点结算
        obs, reward, done = env.step(action_dict)
        if reward < 0:
            print(f"-> ⚠️ Slippage Triggered! Friction Cost: {reward:.2f}")

    print("\n=== Match Ended ===")
    if obs:
        print(f"Final Portfolio: {obs['portfolio']}")

if __name__ == "__main__":
    run_benchmark()