import numpy as np


class WindyGridWorld:
    def __init__(self, epsilon=0.1, alpha=0.5):
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.shape = (7, 10)
        self.start = (3, 0)
        self.finish = (3, 7)
        self.q = np.zeros((*self.shape, 4))
        self.q[self.start] = 0
        self.alpha = alpha
        self.epsilon = epsilon

    def make_move(self, state, action):
        """
        :param state:
        :param action: 0, 1, 2, 3 - up, down, right, left
        :return:
        """
        state_np = np.array(state)
        state_np[0] += self.wind[state[1]]
        if action == 0:
            state_np[0] += 1
        elif action == 1:
            state_np[0] -= 1
        state_np[0] = max(0, min(state_np[0], self.shape[0] - 1))
        if action == 2:
            state_np[1] += 1
        elif action == 3:
            state_np[1] -= 1
        state_np[1] = max(0, min(state_np[1], self.shape[1] - 1))
        return tuple(state_np)

    def pick_action(self, state, greedy=False):
        options = self.q[state]
        probability = np.random.uniform() if not greedy else 1
        return np.argmax(options) if probability >= self.epsilon else np.random.randint(4)

    def make_policy_move(self, state):
        action = self.pick_action(state, greedy=True)
        new_state = self.make_move(state, action)
        return new_state

    def sarsa(self, n_episodes=100):
        for i in range(n_episodes):
            state = self.start
            action = self.pick_action(state)
            while state != self.finish:
                new_state = self.make_move(state, action)
                new_action = self.pick_action(new_state)
                self.q[(*state, action)] += self.alpha * (-1 + self.q[(*new_state, new_action)] - self.q[(*state, action)])
                state = new_state
                action = new_action

    def generate_episode(self):
        all_states = []
        state = self.start
        all_states.append(state)
        while state != self.finish:
            state = self.make_policy_move(state)
            all_states.append(state)
            if len(all_states) > 100:
                print("episode didn't finish after 100 moves, breaking")
                break
        return all_states


if __name__ == "__main__":
    wgw = WindyGridWorld()
    for i in range(1, 21):
        wgw.sarsa(n_episodes=200)
        all_states = wgw.generate_episode()
        print(len(all_states))
    print(all_states)
