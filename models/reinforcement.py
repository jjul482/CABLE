import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
GAMMA = 0.99
CLIP_EPSILON = 0.2
LR = 3e-4
PPO_EPOCHS = 4
FEATURE_DIM = 16
NUM_ADAPTERS = 3
NUM_CLASSES = 10

# Task loss
def task_loss(X, adapter_params):
    return ((X - adapter_params) ** 2).mean()

# Task similarity Λ(𝒳, A)
def task_similarity(X_new, X_old, params_t, params_tp1):
    num = task_loss(X_new, params_tp1) * task_loss(X_old, params_tp1)
    den = task_loss(X_new, params_t) * task_loss(X_old, params_t) + 1e-8
    return 1.0 - num / den

# Reward function
def compute_reward(X_new, y, y_preds, adapters, X_hist):
    reward = 0.0
    for i, adapter in enumerate(adapters):
        sim = task_similarity(X_new, X_hist[i], adapter['t'], adapter['tp1'])
        loss = (y - y_preds[i]) ** 2
        reward += sim * loss
    return reward / (GAMMA * len(adapters))

# Class embedding
def class_embedding(y, num_classes):
    emb = torch.zeros(num_classes)
    emb[y] = 1.0
    return emb

# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(128, output_dim), nn.Softmax(dim=-1))
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

# Advantage estimation
def compute_advantages(rewards, values, gamma=GAMMA):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)

# PPO update
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    for _ in range(PPO_EPOCHS):
        policy, values = model(states)
        dist = Categorical(policy)
        log_probs = dist.log_prob(actions)
        ratio = (log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()

# Setup
state_dim = FEATURE_DIM + NUM_CLASSES + NUM_ADAPTERS
model = ActorCritic(state_dim, NUM_ADAPTERS)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Dummy batch
BATCH_SIZE = 5
X_batch = torch.randn(BATCH_SIZE, FEATURE_DIM)
y_batch = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
y_preds = [torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), dtype=torch.float32) for _ in range(NUM_ADAPTERS)]

adapters = [{
    't': torch.randn(FEATURE_DIM),
    'tp1': torch.randn(FEATURE_DIM),
} for _ in range(NUM_ADAPTERS)]
X_hist = [torch.randn(BATCH_SIZE, FEATURE_DIM) for _ in range(NUM_ADAPTERS)]

states, actions, rewards, values = [], [], [], []

for i in range(BATCH_SIZE):
    y = y_batch[i].item()
    class_emb = class_embedding(y, NUM_CLASSES)
    sims = torch.tensor([
        task_similarity(X_batch[i], X_hist[j][i], adapters[j]['t'], adapters[j]['tp1'])
        for j in range(NUM_ADAPTERS)
    ])
    state = torch.cat([X_batch[i], class_emb, sims])
    states.append(state)

    probs, val = model(state.unsqueeze(0))
    dist = Categorical(probs)
    action = dist.sample()

    reward = compute_reward(X_batch[i], y_batch[i].float(), [y_preds[j][i] for j in range(NUM_ADAPTERS)], adapters, X_hist)

    actions.append(action)
    rewards.append(reward)
    values.append(val.squeeze())

states = torch.stack(states)
actions = torch.stack(actions)
old_log_probs = torch.log(torch.stack([model(s.unsqueeze(0))[0].squeeze(0)[a] for s, a in zip(states, actions)]))
values.append(torch.tensor(0.0))  # bootstrap
returns = torch.tensor([sum(GAMMA**k * rewards[j+k] for k in range(BATCH_SIZE-j)) for j in range(BATCH_SIZE)])
advantages = compute_advantages(rewards, values, GAMMA)

# Train
ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages)