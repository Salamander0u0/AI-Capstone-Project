import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pettingzoo.classic import connect_four_v3


# ---------- Observation handling ----------

def obs_to_tensor(obs_dict):
    board = obs_dict["observation"].reshape(-1)  # 84 features
    mask = obs_dict["action_mask"]               # (7,)
    return (
        torch.tensor(board, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32)
    )


# ---------- DQN network ----------

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(84, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 7)

    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.out(x)
        if mask is not None:
            q = q + (mask == 0) * -1e9
        return q


# ---------- Replay buffer ----------

class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, done = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ---------- Training step ----------

def train_step(policy, target, buffer, optimizer, gamma=0.99, batch_size=256):
    if len(buffer) < batch_size:
        return None

    s, a, r, ns, done = buffer.sample(batch_size)

    q_values = policy(s)
    q = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target(ns)
        next_q = next_q_values.max(1)[0]
        target_q = r + gamma * next_q * (1 - done)

    loss = F.mse_loss(q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


#Human vs AI
def human_vs_ai(policy):
    env = connect_four_v3.env()
    env.reset()

    print("You are player_1 (O). AI is player_0 (X).")
    print("Enter column numbers 0–6.")

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            print("Game over! Reward:", reward)
            env.step(None)
            continue

        board, mask = obs_to_tensor(obs)

        if agent == "player_0":
            # AI move
            with torch.no_grad():
                q = policy(board, mask)
                action = int(torch.argmax(q))
            print(f"AI plays column {action}")

        else:
            # Human move
            legal = torch.where(mask == 1)[0].tolist()
            print("Legal moves:", legal)

            while True:
                move = int(input("Your move: "))
                if move in legal:
                    action = move
                    break
                print("Illegal move. Try again.")

        env.step(action)

# ---------- Main training loop (AEC, correct next-state handling) ----------

def main():
    env = connect_four_v3.env()
    buffer = ReplayBuffer()

    policy = DQN()
    target = DQN()
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    gamma = 0.99
    epsilon = 1.0
    eps_decay = 0.9995
    eps_min = 0.1

    num_episodes = 20000
    target_update_interval = 200

    for episode in range(1, num_episodes + 1):
        env.reset()

        last_state = None
        last_action = None
        episode_reward = 0.0

        # We will store next_state ONLY when player_0 comes around again
        pending_transition = None

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if agent == "player_0":
                episode_reward += reward

            if term or trunc:
                env.step(None)

                # If a transition is pending, finish it with a zero next_state
                if pending_transition is not None:
                    s, a, r = pending_transition
                    ns = torch.zeros_like(s)
                    buffer.push(s, a, r, ns, 1.0)
                    pending_transition = None

                continue

            board_t, mask_t = obs_to_tensor(obs)

            if agent == "player_0":
                # epsilon-greedy
                if random.random() < epsilon:
                    legal_actions = torch.where(mask_t == 1)[0]
                    action = int(random.choice(legal_actions))
                else:
                    with torch.no_grad():
                        q = policy(board_t, mask_t)
                        action = int(torch.argmax(q))

                # If we have a pending transition, complete it now
                if pending_transition is not None:
                    s, a, r = pending_transition
                    buffer.push(s, a, r, board_t, 0.0)
                    pending_transition = None

                # Start a new pending transition
                pending_transition = (board_t, action, reward)

                last_state = board_t
                last_action = action

            else:
                # Opponent: random legal move
                legal_actions = torch.where(mask_t == 1)[0]
                action = int(random.choice(legal_actions))

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
    human_vs_ai(policy)


if __name__ == "__main__":
    main()
    
