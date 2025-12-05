import sys
import os
import time
import torch
import numpy as np

# 确保能找到 src (假设在项目根目录运行)
sys.path.append(os.getcwd())

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# --- 1. 导入你的 Agent (NaiveAgent) ---
# 根据你的描述，Agent 在 agents/Group23/NaiveAgent.py
try:
    from agents.Group23.NaiveAgent import NaiveAgent
except ImportError:
    # 如果类名是 Agent 而不是 NaiveAgent，尝试这个
    try:
        from agents.Group23.NaiveAgent import Agent as NaiveAgent
    except ImportError:
        print("错误: 无法导入 NaiveAgent。请检查 agents/Group23/NaiveAgent.py 文件路径和类名。")
        sys.exit(1)

from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor

# --- 配置 ---
NUM_GAMES = 50          # 生成局数 (建议 50 用于测试，500 用于正式训练)
OUTPUT_FILE = "data/mcts_advanced_games.pt"

def generate_data():
    print(f"--- 开始生成数据: {NUM_GAMES} 局 (MCTS Visits + Winner) ---")
    print("正在初始化 Agent...")
    
    all_data_buffer = [] 
    start_time = time.time()
    
    # 初始化两个 Agent
    agent_red = NaiveAgent(Colour.RED)
    agent_blue = NaiveAgent(Colour.BLUE)

    for i in range(NUM_GAMES):
        game_start_time = time.time()
        # print(f"\n[Game {i+1}/{NUM_GAMES}] 开始...")
        
        board = Board(11)
        turn = 1
        curr_colour = Colour.RED
        
        # 暂存本局数据: [(board_tensor, visit_probs, player), ...]
        game_history = [] 
        winner_colour = None
        move_count = 0

        while True:
            current_agent = agent_red if curr_colour == Colour.RED else agent_blue
            
            # 1. 让 Agent 思考 (make_move 会运行 MCTS)
            # 如果你的 Agent 支持 time_limit 参数，可以在这里加上
            move = current_agent.make_move(turn, board, None)
            
            # 记录数据 (跳过 Swap)
            if move.x != -1:
                # A. 棋盘状态 Tensor (C, H, W)
                tensor = encode_board_to_tensor(board, curr_colour).squeeze(0)
                
                # B. 获取 MCTS 访问分布 (Policy Target)
                # 我们想要一个 121 维的概率分布，而不仅仅是 One-Hot
                visits_map = torch.zeros(11 * 11, dtype=torch.float32)
                
                try:
                    # 尝试读取 Agent 内部的 MCTS 树
                    # 假设结构是 agent.mcts.root.children
                    if hasattr(current_agent, 'mcts') and current_agent.mcts.root is not None:
                        root_node = current_agent.mcts.root
                        total_visits = 0
                        
                        for child in root_node.children:
                            # child.move 应该是 (row, col)
                            r, c = child.move
                            count = child.n_visits
                            
                            idx = r * 11 + c
                            visits_map[idx] = count
                            total_visits += count
                        
                        # 归一化为概率 (Probability Distribution)
                        if total_visits > 0:
                            visits_map /= total_visits
                    else:
                        # 如果读不到内部数据，回退到 One-Hot (只标记最终走的那一步)
                        # 这依然是有用的数据，只是不如 Visits 分布丰富
                        idx = move.x * 11 + move.y
                        visits_map[idx] = 1.0
                except Exception:
                    # 发生任何错误都回退到 One-Hot，保证程序不崩
                    idx = move.x * 11 + move.y
                    visits_map[idx] = 1.0

                # 存入历史: (Board, Policy, Player)
                game_history.append([tensor, visits_map, curr_colour])
                
                # 执行移动
                board.set_tile_colour(move.x, move.y, curr_colour)
            
            # 检查胜负
            if board.has_ended(curr_colour):
                winner_colour = curr_colour
                break
                
            curr_colour = Colour.opposite(curr_colour)
            turn += 1
            move_count += 1
            if turn > 121: break # 平局

        # --- 游戏结束，回填输赢信息 (Value Target) ---
        # 赢家 = +1.0, 输家 = -1.0, 平局 = 0.0
        for step_data in game_history:
            board_tensor, policy_target, player = step_data
            
            if winner_colour is None:
                value_target = 0.0
            elif player == winner_colour:
                value_target = 1.0
            else:
                value_target = -1.0
                
            # 将 (Board, Policy, Value) 加入总数据集
            all_data_buffer.append((board_tensor, policy_target, torch.tensor(value_target, dtype=torch.float32)))

        # 打印进度 (每1局打印一次，让你知道它活着)
        game_duration = time.time() - game_start_time
        print(f"进度: {i + 1}/{NUM_GAMES} 局 | 步数: {move_count} | 耗时: {game_duration:.1f}s | 累计样本: {len(all_data_buffer)}")
        
        # 定期保存 (每10局)，防止跑一半断电
        if (i + 1) % 10 == 0:
            if not os.path.exists("data"): os.makedirs("data")
            torch.save(all_data_buffer, OUTPUT_FILE)

    # 最终保存
    if not os.path.exists("data"): os.makedirs("data")
    torch.save(all_data_buffer, OUTPUT_FILE)
    
    total_time = time.time() - start_time
    print(f"\n--- 数据生成完毕! 总耗时: {total_time/60:.1f} 分钟 ---")
    print(f"--- 数据已保存至: {OUTPUT_FILE} ---")

if __name__ == "__main__":
    generate_data()