import os
import math
import random
import multiprocessing as mp
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from pettingzoo.classic import connect_four_v3

# Gemini explanation imports
from explanation import build_explanation_payload, explain_move_with_gemini

# ============================================================
# CONFIG (v4)
# ============================================================

BOARD_ROWS = 6
BOARD_COLS = 7

MCTS_SIM_TRAIN = 256
MCTS_SIM_PLAY = 384

MAX_BUFFER_SIZE = 150000

GATING_GAMES = 60
GATING_THRESHOLD = 0.52

NUM_ITERATIONS = 250
GAMES_PER_ITER = 48
BATCH_SIZE = 128


# ============================================================
# NETWORK (Policy + Value)
# ============================================================

class AZNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        self.fc = nn.Linear(128 * 1 * 2, 256)

        self.policy_head = nn.Linear(256, BOARD_COLS)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x, mask=None):
        if x.dim() == 1:
            x = x.view(1, 2, BOARD_ROWS, BOARD_COLS)
        else:
            x = x.view(-1, 2, BOARD_ROWS, BOARD_COLS)

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)

        if mask is not None:
            policy_logits = policy_logits + (mask == 0) * -1e9

        return policy_logits, value


# ============================================================
# GAME STATE
# ============================================================

class ConnectFourState:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.current_player = 1

    def clone(self):
        s = ConnectFourState()
        s.board = [row[:] for row in self.board]
        s.current_player = self.current_player
        return s

    def legal_moves(self):
        return [c for c in range(BOARD_COLS) if self.board[0][c] == 0]

    def apply_move(self, col):
        for r in range(BOARD_ROWS - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.current_player
                break
        self.current_player *= -1

    def check_winner(self):
        b = self.board
        # horizontal
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                line = [b[r][c + i] for i in range(4)]
                if line[0] != 0 and all(v == line[0] for v in line):
                    return line[0]
        # vertical
        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 3):
                line = [b[r + i][c] for i in range(4)]
                if line[0] != 0 and all(v == line[0] for v in line):
                    return line[0]
        # diag down-right
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                line = [b[r + i][c + i] for i in range(4)]
                if line[0] != 0 and all(v == line[0] for v in line):
                    return line[0]
        # diag up-right
        for r in range(3, BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                line = [b[r - i][c + i] for i in range(4)]
                if line[0] != 0 and all(v == line[0] for v in line):
                    return line[0]
        # draw
        if all(b[0][c] != 0 for c in range(BOARD_COLS)):
            return None
        return 0

    def is_terminal(self):
        return self.check_winner() != 0

    def to_tensor(self):
        cur = [[1 if self.board[r][c] == self.current_player else 0
                for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]
        opp = [[1 if self.board[r][c] == -self.current_player else 0
                for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]
        planes = [cur, opp]
        return torch.tensor(planes, dtype=torch.float32)


# ============================================================
# REPLAY BUFFER
# ============================================================

class AZReplayBuffer:
    def __init__(self, size=MAX_BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def push(self, s, pi, v):
        self.buffer.append((s, pi, v))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pi, v = zip(*batch)
        return (
            torch.stack(s),
            torch.stack(pi),
            torch.tensor(v, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# MCTS NODE
# ============================================================

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ============================================================
# CHECK WIN AT POSITION
# ============================================================

def check_win_at(board, row, col, player):

    # Horizontal
    count = 0
    for c in range(max(0, col - 3), min(BOARD_COLS, col + 4)):
        if board[row][c] == player:
            count += 1
            if count >= 4:
                return True
        else:
            count = 0

    # Vertical
    count = 0
    for r in range(max(0, row - 3), min(BOARD_ROWS, row + 4)):
        if board[r][col] == player:
            count += 1
            if count >= 4:
                return True
        else:
            count = 0

    # Diagonal down-right
    count = 0
    start_r = row - min(3, row)
    start_c = col - min(3, col)
    r, c = start_r, start_c
    while r < BOARD_ROWS and c < BOARD_COLS:
        if board[r][c] == player:
            count += 1
            if count >= 4:
                return True
        else:
            count = 0
        r += 1
        c += 1

    # Diagonal up-right
    count = 0
    start_r = row + min(3, BOARD_ROWS - 1 - row)
    start_c = col - min(3, col)
    r, c = start_r, start_c
    while r >= 0 and c < BOARD_COLS:
        if board[r][c] == player:
            count += 1
            if count >= 4:
                return True
        else:
            count = 0
        r -= 1
        c += 1

    return False


# ============================================================
# FIND WINNING MOVE
# ============================================================

def find_winning_move(state, player):

    for col in state.legal_moves():
        row = None
        for r in range(BOARD_ROWS - 1, -1, -1):
            if state.board[r][col] == 0:
                row = r
                break

        if row is None:
            continue

        state.board[row][col] = player

        if check_win_at(state.board, row, col, player):
            state.board[row][col] = 0
            return col

        state.board[row][col] = 0

    return None


# ============================================================
# MCTS SEARCH (UPDATED TO RETURN pi, visits)
# ============================================================

def mcts_search(root_state, net, num_simulations):
    root = MCTSNode(root_state)

    # 1. Immediate win
    win_move = find_winning_move(root_state, root_state.current_player)
    if win_move is not None:
        pi = torch.zeros(BOARD_COLS)
        pi[win_move] = 1.0
        visits = [1 if i == win_move else 0 for i in range(BOARD_COLS)]
        return pi, visits

    # 2. Immediate block
    opp = -root_state.current_player
    block_move = find_winning_move(root_state, opp)
    if block_move is not None:
        pi = torch.zeros(BOARD_COLS)
        pi[block_move] = 1.0
        visits = [1 if i == block_move else 0 for i in range(BOARD_COLS)]
        return pi, visits

    with torch.no_grad():
        s_tensor = root_state.to_tensor()
        mask = torch.zeros(BOARD_COLS)
        for a in root_state.legal_moves():
            mask[a] = 1
        logits, value = net(s_tensor, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    for a in root_state.legal_moves():
        child_state = root_state.clone()
        child_state.apply_move(a)
        root.children[a] = MCTSNode(child_state, parent=root, prior=probs[a].item())

    c_puct = 1.5

    for _ in range(num_simulations):
        node = root

        # selection
        while node.children:
            best_score = -1e9
            best_action = None
            for a, child in node.children.items():
                u = c_puct * child.prior * math.sqrt(node.visit_count + 1) / (child.visit_count + 1)
                score = child.value + u
                if score > best_score:
                    best_score = score
                    best_action = a
            node = node.children[best_action]

        state = node.state
        winner = state.check_winner()
        if winner == 0:
            with torch.no_grad():
                s_tensor = state.to_tensor()
                mask = torch.zeros(BOARD_COLS)
                for a in state.legal_moves():
                    mask[a] = 1
                logits, value = net(s_tensor, mask)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
            leaf_value = value.item()
            node.children = {}
            for a in state.legal_moves():
                child_state = state.clone()
                child_state.apply_move(a)
                node.children[a] = MCTSNode(child_state, parent=node, prior=probs[a].item())
        else:
            if winner is None:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == root_state.current_player else -1.0

        v = leaf_value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            node = node.parent

    pi = torch.zeros(BOARD_COLS)
    visits = [0] * BOARD_COLS
    for a, child in root.children.items():
        pi[a] = child.visit_count
        visits[a] = child.visit_count

    if pi.sum() > 0:
        pi = pi / pi.sum()

    return pi, visits

# ============================================================
# TEMPERATURE SCHEDULE
# ============================================================

def temperature_for_move(move_index):
    if move_index < 8:
        return 1.0
    elif move_index < 16:
        return 0.5
    else:
        return 0.1


def apply_temperature(pi, legal_moves, move_index):
    T = temperature_for_move(move_index)
    if T == 0.0:
        return pi

    legal_probs = pi[legal_moves]
    legal_probs = legal_probs ** (1.0 / T)

    if legal_probs.sum() <= 0:
        legal_probs = torch.ones_like(legal_probs)

    legal_probs = legal_probs / legal_probs.sum()

    out = torch.zeros_like(pi)
    out[legal_moves] = legal_probs
    return out


# ============================================================
# SELF-PLAY WORKER
# ============================================================

def selfplay_worker(worker_id, net_state_dict, num_games, queue):
    net = AZNet()
    net.load_state_dict(net_state_dict)
    net.eval()

    for _ in range(num_games):
        state = ConnectFourState()
        states = []
        policies = []
        players = []

        move_index = 0

        while not state.is_terminal():
            pi, _ = mcts_search(state.clone(), net, num_simulations=MCTS_SIM_TRAIN)

            legal = state.legal_moves()
            pi_temp = apply_temperature(pi, legal, move_index)

            states.append(state.to_tensor())
            policies.append(pi_temp)
            players.append(state.current_player)

            probs = pi_temp[legal]
            probs = probs / probs.sum()
            action = random.choices(legal, weights=probs.tolist(), k=1)[0]
            state.apply_move(action)

            move_index += 1

        winner = state.check_winner()
        if winner is None:
            final_reward = 0.0
        else:
            final_reward = 1.0 if winner == 1 else -1.0

        for s, pi, p in zip(states, policies, players):
            v = final_reward * p
            queue.put((s, pi, v))

            s_flip = torch.flip(s, dims=[2])
            pi_flip = torch.flip(pi, dims=[0])
            queue.put((s_flip, pi_flip, v))


# ============================================================
# GATING EVALUATION
# ============================================================

def play_head_to_head(best_net, cand_net, num_games):
    best_net.eval()
    cand_net.eval()

    cand_wins = 0
    best_wins = 0
    draws = 0

    for g in range(num_games):
        state = ConnectFourState()
        cand_is_player1 = (g % 2 == 0)
        move_index = 0

        while not state.is_terminal():
            if (state.current_player == 1 and cand_is_player1) or (state.current_player == -1 and not cand_is_player1):
                pi, _ = mcts_search(state.clone(), cand_net, num_simulations=MCTS_SIM_TRAIN)
            else:
                pi, _ = mcts_search(state.clone(), best_net, num_simulations=MCTS_SIM_TRAIN)

            legal = state.legal_moves()
            pi_temp = apply_temperature(pi, legal, move_index)

            probs = pi_temp[legal]
            probs = probs / probs.sum()
            action = random.choices(legal, weights=probs.tolist(), k=1)[0]
            state.apply_move(action)

            move_index += 1

        winner = state.check_winner()
        if winner is None:
            draws += 1
        else:
            if (winner == 1 and cand_is_player1) or (winner == -1 and not cand_is_player1):
                cand_wins += 1
            else:
                best_wins += 1

    total = cand_wins + best_wins + draws
    cand_rate = cand_wins / total if total > 0 else 0.0
    return cand_rate, cand_wins, best_wins, draws


# ============================================================
# GATED TRAINING LOOP (v4)
# ============================================================

def train_mcts_gated_v3():
    os.makedirs("models", exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    best_net = AZNet()

    if os.path.exists("models/connect4_mcts_v4_final.pth"):
        best_net.load_state_dict(torch.load("models/connect4_mcts_v4_final.pth"))
        print("Loaded existing v4 best model.")
    elif os.path.exists("models/connect4_mcts_v3_final.pth"):
        best_net.load_state_dict(torch.load("models/connect4_mcts_v3_final.pth"))
        print("Loaded existing v3 best model.")
    else:
        print("Starting v4 training from scratch.")

    buffer = AZReplayBuffer(size=MAX_BUFFER_SIZE)

    metrics_path = "metrics_mcts_v4.csv"
    with open(metrics_path, "w") as f:
        f.write("iteration,buffer_size,loss,cand_rate,cand_wins,best_wins,draws\n")

    num_cpus = os.cpu_count() or 4
    num_workers = min(8, num_cpus)

    for it in tqdm(range(1, NUM_ITERATIONS + 1), desc="Gated MCTS v4 training"):
        manager = mp.Manager()
        queue = manager.Queue()

        best_state_dict = best_net.state_dict()
        games_per_worker = math.ceil(GAMES_PER_ITER / num_workers)

        workers = []
        for wid in range(num_workers):
            p = mp.Process(
                target=selfplay_worker,
                args=(wid, best_state_dict, games_per_worker, queue)
            )
            p.start()
            workers.append(p)

        for p in workers:
            p.join()

        while not queue.empty():
            s, pi, v = queue.get()
            buffer.push(s, pi, v)

        cand_net = AZNet()
        cand_net.load_state_dict(best_net.state_dict())
        optimizer = optim.Adam(cand_net.parameters(), lr=1e-4)

        avg_loss = None

        if len(buffer) >= BATCH_SIZE:
            losses = []
            for _ in range(100):
                s_batch, pi_batch, v_batch = buffer.sample(BATCH_SIZE)
                logits, values = cand_net(s_batch)

                log_probs = torch.log_softmax(logits, dim=-1)
                policy_loss = -(pi_batch * log_probs).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(values, v_batch)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)

        cand_rate, cand_wins, best_wins, draws = play_head_to_head(best_net, cand_net, GATING_GAMES)

        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"

        print(
            f"Iter {it} | buffer={len(buffer)} | loss={loss_str} | "
            f"cand_rate={cand_rate:.3f} (W:{cand_wins}, L:{best_wins}, D:{draws})"
        )

        with open(metrics_path, "a") as f:
            f.write(f"{it},{len(buffer)},{loss_str},{cand_rate},{cand_wins},{best_wins},{draws}\n")

        if cand_rate >= GATING_THRESHOLD:
            best_net.load_state_dict(cand_net.state_dict())
            torch.save(best_net.state_dict(), "models/connect4_mcts_v4_final.pth")
            print(f"New v4 best model accepted at iteration {it} (rate={cand_rate:.3f})")

        if it % 25 == 0:
            torch.save(best_net.state_dict(), f"models/connect4_mcts_v4_iter{it}.pth")

    torch.save(best_net.state_dict(), "models/connect4_mcts_v4_final.pth")
    print("Gated MCTS v4 training complete.")


# ============================================================
# HUMAN VS AI (with Gemini explanations)
# ============================================================

def human_vs_ai_v3(model_path="models/connect4_mcts_v4_final.pth"):
    net = AZNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    env = connect_four_v3.env(render_mode="human")
    env.reset()

    state = ConnectFourState()

    print("You are player_1 (O). AI is player_0 (X).")

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            print("Game over! Reward:", reward)
            env.step(None)
            continue

        if agent == "player_0":
            pi, visits = mcts_search(state.clone(), net, num_simulations=MCTS_SIM_PLAY)

            legal = state.legal_moves()
            pi_temp = apply_temperature(pi, legal, move_index=0)
            probs = pi_temp[legal]
            probs = probs / probs.sum()
            action = random.choices(legal, weights=probs.tolist(), k=1)[0]

            # ---------------------------------------------------------
            # Gemini Explanation
            # ---------------------------------------------------------
            payload = build_explanation_payload(
                state.clone(),
                pi,
                visits,
                action,
                find_winning_move
            )

            #explanation = explain_move_with_gemini(payload)
            #print("\nAI explanation:")
            #print(explanation)
            # ---------------------------------------------------------

            print(f"AI plays column {action}")
            state.apply_move(action)

        else:
            legal = state.legal_moves()
            print("Legal moves:", legal)

            while True:
                move = int(input("Your move: "))
                if move in legal:
                    action = move
                    break
                print("Illegal move.")
            state.apply_move(action)

        env.step(action)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # To train v4:
    #train_mcts_gated_v3()

    # To play against v4:
    human_vs_ai_v3("models/connect4_mcts_v4_final.pth")
    #human_vs_ai_v3("models/connect4_mcts_v4_iter250.pth")
    
