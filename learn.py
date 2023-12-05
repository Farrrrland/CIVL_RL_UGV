import numpy as np
from path_env import PathEnv

if __name__ == "__main__":
    env = PathEnv()
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