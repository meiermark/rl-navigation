#!/usr/bin/env python


class NStepBuffer:
    def __init__(self, step_size, gamma):
        self.gamma = gamma
        self.step_size = step_size
        self.episode = []

    def append(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward, next_state, done))

    def get_all(self):
        proc_episode = []

        for n, t in enumerate(self.episode):
            reward = t[2]
            for i in range(2, self.step_size+1):
                if n+i-1 >= len(self.episode):
                    break
                reward += pow(self.gamma, i-1) * self.episode[n+i-1][2]
            proc_episode.append((t[0], t[1], reward, t[3], t[4]))

        self.episode = []
        return proc_episode
