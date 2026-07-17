import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pettingzoo.classic import connect_four_v3


# ---------- Dueling CNN Double-DQN Model ----------

class DuelingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # After convs: (2,6,7) -> (32,3,4) -> (64,1,2) -> 128 features
        self.fc = nn.Linear(64 * 1 * 2, 128)

        # Dueling streams
        self.value_fc = nn.Linear(128, 1)
        self.adv_fc = nn.Linear(128, 7)

    def forward(self, x, mask=None):
        if x.dim() == 1:
            x = x.view(1, 2, 6, 7)
        else:
            x = x.view(-1, 2, 6, 7)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        value = self.value_fc(x)          # (batch, 1)
        adv = self.adv_fc(x)             # (batch, 7)

        # Dueling combination
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = value + (adv - adv_mean)

        if mask is not None:
            q = q + (mask == 0) * -1e9

        return q


# ---------- Prioritized Replay Buffer ----------

class PrioritizedReplayBuffer:
    def __init__(self, size=50000, alpha=0.6):
        self.size = size
        self.alpha = alpha
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    def push(self, s, a, r, ns, done, td_error=1.0):
        self.buffer.append((s, a, r, ns, done))
        # priority = (|td_error| + eps)^alpha
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None

        priorities = list(self.priorities)
        probs = [p / sum(priorities) for p in priorities]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)

        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, done = zip(*batch)

        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(done, dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(err.item()) + 1e-5) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# ---------- Observation Handling ----------

def obs_to_tensor(obs_dict):
    board = obs_dict["observation"]  # (6,7,2)
    board = board.transpose(2, 0, 1)  # -> (2,6,7)
    board_t = torch.tensor(board, dtype=torch.float32)
    mask_t = torch.tensor(obs_dict["action_mask"], dtype=torch.float32)
    return board_t, mask_t


# ---------- Double-DQN Training Step ----------

def train_step(policy, target, buffer, optimizer, gamma=0.99, batch_size=256):
    if len(buffer) < batch_size:
        return None

    sample = buffer.sample(batch_size)
    if sample is None:
        return None

    s, a, r, ns, done, indices = sample

    q_values = policy(s)
    q = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double-DQN: action from policy, value from target
        next_q_policy = policy(ns)
        next_actions = next_q_policy.argmax(dim=1)

        next_q_target = target(ns)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q = r + gamma * next_q * (1 - done)

    td_errors = target_q - q
    loss = F.mse_loss(q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities
    buffer.update_priorities(indices, td_errors)

    return loss.item()


# ---------- Evaluate Win-Rate vs Opponent Snapshot ----------

def evaluate_winrate(policy, opponent_path, num_games=50):
    if opponent_path is None or not os.path.exists(opponent_path):
        return None

    opponent = DuelingCNN()
    opponent.load_state_dict(torch.load(opponent_path))
    opponent.eval()

    env = connect_four_v3.env()
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        env.reset()
        pending_transition = None

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                env.step(None)
                if agent == "player_0":
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        draws += 1
                continue

            board_t, mask_t = obs_to_tensor(obs)

            if agent == "player_0":
                with torch.no_grad():
                    q = policy(board_t, mask_t)
                    action = int(torch.argmax(q))
            else:
                with torch.no_grad():
                    q = opponent(board_t, mask_t)
                    action = int(torch.argmax(q))

            env.step(action)

    win_rate = wins / num_games
    return win_rate


# ---------- Self-Play Training Loop with Snapshots and Win-Rate Tracking ----------

def train_selfplay():
    env = connect_four_v3.env()
    buffer = PrioritizedReplayBuffer(size=50000, alpha=0.6)

    policy = DuelingCNN()
    target = DuelingCNN()
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    gamma = 0.99
    epsilon = 1.0
    eps_decay = 0.9995
    eps_min = 0.1

    num_episodes = 20000
    target_update_interval = 200
    save_interval = 1000

    os.makedirs("models", exist_ok=True)

    metrics_path = "metrics_winrate.csv"
    with open(metrics_path, "w") as f:
        f.write("episode,win_rate\n")

    last_snapshot_path = None

    for episode in range(1, num_episodes + 1):
        env.reset()
        pending_transition = None
        episode_reward = 0.0

        # Choose opponent model for this episode (snapshot or current)
        opponent_model = DuelingCNN()
        if last_snapshot_path is not None and os.path.exists(last_snapshot_path):
            opponent_model.load_state_dict(torch.load(last_snapshot_path))
        else:
            opponent_model.load_state_dict(policy.state_dict())
        opponent_model.eval()

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if agent == "player_0":
                episode_reward += reward

            if term or trunc:
                env.step(None)

                if pending_transition is not None:
                    s, a, r = pending_transition
                    ns = torch.zeros_like(s)
                    buffer.push(s, a, r, ns, 1.0)
                    pending_transition = None

                continue

            board_t, mask_t = obs_to_tensor(obs)

            if agent == "player_0":
                if random.random() < epsilon:
                    legal = torch.where(mask_t == 1)[0]
                    action = int(random.choice(legal))
                else:
                    with torch.no_grad():
                        q = policy(board_t, mask_t)
                        action = int(torch.argmax(q))

                if pending_transition is not None:
                    s, a, r = pending_transition
                    buffer.push(s, a, r, board_t, 0.0)
                    pending_transition = None

                pending_transition = (board_t, action, reward)

            else:
                with torch.no_grad():
                    q = opponent_model(board_t, mask_t)
                    action = int(torch.argmax(q))

            env.step(action)

        epsilon = max(eps_min, epsilon * eps_decay)

        loss = train_step(policy, target, buffer, optimizer, gamma=gamma)

        if episode % target_update_interval == 0:
            target.load_state_dict(policy.state_dict())

        if episode % 100 == 0:
            print(
                f"Episode {episode} | "
                f"epsilon={epsilon:.3f} | "
                f"reward={episode_reward:.2f} | "
                f"loss={loss if loss is not None else 'N/A'}"
            )

        if episode % save_interval == 0:
            snapshot_path = f"models/connect4_v{episode}.pth"
            torch.save(policy.state_dict(), snapshot_path)
            last_snapshot_path = snapshot_path
            print(f"Saved snapshot: {snapshot_path}")

            win_rate = evaluate_winrate(policy, last_snapshot_path, num_games=50)
            if win_rate is not None:
                with open(metrics_path, "a") as f:
                    f.write(f"{episode},{win_rate}\n")
                print(f"Episode {episode} | win_rate vs snapshot: {win_rate:.3f}")

    latest_path = "models/connect4_latest.pth"
    torch.save(policy.state_dict(), latest_path)
    print(f"Training complete. Latest model saved to {latest_path}")


# ---------- Human vs AI Gameplay ----------

def human_vs_ai(model_path):
    policy = DuelingCNN()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    env = connect_four_v3.env(render_mode="human")
    env.reset()

    print("You are player_1 (O). AI is player_0 (X).")

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            print("Game over! Reward:", reward)
            env.step(None)
            continue

        board_t, mask_t = obs_to_tensor(obs)

        if agent == "player_0":
            with torch.no_grad():
                q = policy(board_t, mask_t)
                action = int(torch.argmax(q))
            print(f"AI plays column {action}")
        else:
            legal = torch.where(mask_t == 1)[0].tolist()
            print("Legal moves:", legal)

            while True:
                move = int(input("Your move: "))
                if move in legal:
                    action = move
                    break
                print("Illegal move.")

        env.step(action)


# ---------- Entry Point ----------

if __name__ == "__main__":
    # Uncomment ONE of these:

    # 1) Train with self-play, snapshots, Double-DQN, dueling CNN, prioritized replay
    #train_selfplay()

    # 2) Play against a saved model (e.g., latest or a snapshot)
    #human_vs_ai("models/connect4_latest.pth")
    human_vs_ai("models/connect4_v6000.pth")