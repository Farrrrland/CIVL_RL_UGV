import gym
from gym import spaces
import numpy as np

class PathEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Discrete(10)
        self.adjacent = [
            [0, 81, -1, -1, -1, 81, -1, -1, -1, -1],
            [81, 0, 81, -1, -1, -1, 81, -1, -1, -1],
            [-1, 81, 0, 40, -1, -1, -1, -1, -1, -1],
            [-1, -1, 40, 0, 81, -1, -1, 41, -1, -1],
            [-1, -1, -1, 81, 0, -1, -1, -1, 81, -1],
            [81, -1, -1, -1, -1, 0, 81, -1, -1, -1],
            [-1, 81, -1, -1, -1, 81, 0, 81, -1, -1],
            [-1, -1, -1, 41, -1, -1, 81, 0, -1, -1],
            [-1, -1, -1, -1, 81, -1, -1, -1, 0, 81],
            [-1, -1, -1, -1, -1, -1, -1, -1, 81, 0]
        ]
        self.smoke = [1, 2, 6, 7, 8, 3, 4, 5, 9, 10]
        self.score = 0
        self.state = 0
        for adj in self.adjacent:
            for d in adj:
                if d > 0:
                    self.score += d
        print(self.score)

    def step(self, action):
        done = False
        Info = {}
        Info['state'] = self.state
        Info['action'] = action
        prev_state = self.state
        # Smoke detection
        obs = []
        for idx in range(10):
            smk = 0
            if self.adjacent[action][idx] > 0:
                smk = self.smoke[idx]
            obs.append(smk)
        
        if action == self.state or self.adjacent[self.state][action] < 0:
            done = True
            reward = -self.score
            Info['next_state'] = -1
            Info['reward'] = reward
            Info['done'] = True
            Info['info'] = "Hit the wall!"
            return -1, reward, done, Info
        else:
            smk_current = self.smoke[action]
            reward = -self.adjacent[self.state][action] + smk_current * 3
            self.state = action
            Info['next_state'] = self.state
            if smk_current >= np.max(obs):
                done = True
                reward += self.score
                Info['reward'] = reward
                Info['done'] = True
                Info['episodic_return'] = reward
                Info['info'] = "Arrival"
                return self.state, reward, done, Info
            else:
                Info['reward'] = reward
                Info['done'] = False
                Info['info'] = "Step"
                return self.state, reward, done, Info
    
    def reset(self):
        self.state = 0
        return self.state
    
    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

def smoothed(interval, window_length, k):
    return scipy.signal.savgol_filter(interval, window_length, k)