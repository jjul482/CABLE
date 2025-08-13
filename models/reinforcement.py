import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ClippedPPO:
    def __init__(
        self,
        adapters,
        loss_fn,
        batch,
        state_dim=None,
        action_dim=None,
        clip_epsilon=0.2,
        gamma=0.99,
        lr=1e-3,
        device="cpu"
    ):
        self.adapters = adapters
        self.loss_fn = loss_fn
        self.batch = batch
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.device = device

        # Policy network: maps state to action probabilities
        self.state_dim = state_dim or len(batch)
        self.action_dim = action_dim or len(adapters)
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.old_log_probs = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.old_log_probs.append(log_prob.detach())
        return action.item()

    def assign_class_to_adapter(self, action):
        if action < len(self.adapters):
            assigned_adapter = self.adapters[action]
            assigned_adapter.assign_batch(self.batch)
            return assigned_adapter
        else:
            raise ValueError("Action out of bounds for available adapters.")

    def receive_reward(self, reward):
        self.rewards.append(reward)

    def finish_episode(self):
        # Compute discounted rewards
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.old_log_probs).detach()

        # PPO update
        for _ in range(4):  # K epochs
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * returns
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.old_log_probs = []

    def update_policy(self):
        self.finish_episode()