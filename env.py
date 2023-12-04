import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MySim(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(6)
        self.adjacent = [[0, 1, 12, -1, -1, -1], 
                    [-1, 0, 9, 3, -1, -1], 
                    [-1, -1, 0, -1, 5, -1], 
                    [-1, -1, 4, 0, 13, 15], 
                    [-1, -1, -1, -1, 0, 4], 
                    [-1, -1, -1, -1, -1, 0]]
        self.score = 0
        self.state = 0
        for adj in self.adjacent:
            for d in adj:
                if d > 0:
                    self.score += d
        # self.reward = 0
        # print(self.reward)

    def test(self):
        for adj in self.adjacent:
            print(adj)

    def step(self, action):
        # print(f"From {self.state} to {action}, dist {self.adjacent[self.state][action]}")
        done = False
        Info = {}
        Info['state'] = self.state
        Info['action'] = action
        prev_state = self.state
        if action == self.state or self.adjacent[self.state][action] < 0:
            done = True
            reward = -self.score
            Info['next_state'] = -1
            Info['reward'] = reward
            Info['done'] = True
            Info['info'] = "Hit the wall!"
            return -1, reward, done, Info
        else:
            reward = -self.adjacent[self.state][action]
            self.state = action
            Info['next_state'] = self.state
            if self.state == 5:
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

if __name__ == "__main__":
    env = MySim()
    env.test()
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    print(Q)
    lr , gamma, num_episodes, epsi = .5, .99, 2000, .9
    rList, jList = [], []

    for i in range(num_episodes):
        traj = []
        s = env.reset()
        traj.append(s)

        rAll, d, j = 0, False, 0
        while j < 99:
            j += 1
            if np.random.rand() > epsi and i < 600:
                print("RAND")
                a = np.random.randint(0, env.action_space.n)
            else:
                a = np.argmax(Q[s, :])

            s1, r, d, inf = env.step(a)
            traj.append(s1)
            Q[s, a] = Q[s, a] + lr * (r + gamma * np.max(Q[s1, : ]) - Q[s, a])
            rAll += r
            if 'episodic_return' in inf.keys():
                print(traj)
            s = s1
            if d == True:
                break
        jList.append(j)
        rList.append(rAll)
        print(f"Episode [{i + 1}/{num_episodes}], Score: {rAll}, Length: {j}, Score avg: {format(sum(rList)/(i+1), '.4f')}")
    # print(Q)


fig, ax = plt.subplots(figsize=(10, 6.5), dpi=500)

ax.plot(range(len(rList)), rList, color=sns.color_palette("Purples")[2], lw=2)

ax.set_xlabel("Update Time (#)", size = 28)
ax.set_ylabel("Episodic Return", size = 28)
plt.yticks(size = 18)
plt.xticks(size = 18)
# ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig(f"test-shortest-path.pdf")