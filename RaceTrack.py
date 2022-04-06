import numpy as np
import random


class RaceTrack:
    def __init__(self, simple=True, gamma=0.9):
        if simple:
            self.field = np.zeros((32, 17), dtype=int)
            self.field[0, 3:] = 1
            self.field[1, 2:] = 1
            self.field[2, 2:] = 1
            self.field[3, 1:] = 1
            self.field[4, :] = 1
            self.field[5, :] = 1
            self.field[6, :10] = 1
            self.field[7:14, :9] = 1
            self.field[14:22, 1:9] = 1
            self.field[22:29, 2:9] = 1
            self.field[29:, 3:9] = 1
            self.field = np.flipud(self.field)
            self.finish_line = 16
        else:
            raise Exception('not supported')

        # state is position on the field and current speed, so, to encode states, you need four dimensions.
        # action is changing velocity components by +1, 0, -1 each, so, another two dimensions?
        # policy is action value per state, so it needs to be decided how to store chosen actions.
        # Put ones in action array?
        # action-value function
        self.q = np.zeros((*self.field.shape, 5, 5, 3, 3))
        # cumulative sum of weights of returns
        self.c = np.zeros((*self.field.shape, 5, 5, 3, 3))
        self.policy = np.zeros((*self.field.shape, 5, 5, 3, 3))
        self.policy[:, :, :, :, 1, 1] = 1
        self.random_policy = np.ones((*self.field.shape, 5, 5, 3, 3)) / 9
        self.gamma = gamma

    @staticmethod
    def sample_index(p):
        i = np.random.choice(np.arange(p.size), p=p.ravel())
        return np.array(np.unravel_index(i, p.shape))

    @staticmethod
    def find_all_crossed_cells(start, finish):
        """
        :param start: [a, b]
        :param finish: [c, d]
        :return: list of cells/indices that this segment crosses
        """
        return np.unique(i.astype(int) for i in np.linspace(start, finish, 1000))

    def check_boundary_cross(self, start, finish):
        """
        :param start:
        :param finish:
        :return: 0, if didn't cross; 1, if crossed boundary but not finish, 2, if crossed at finish line
        """
        crossed_cells = self.find_all_crossed_cells(start, finish)
        prev_cell = crossed_cells[0]
        for cc in crossed_cells[1:]:
            if cc[0] >= self.field.shape[0] or cc[1] >= self.field.shape[1] or self.field[cc] == 0:
                if prev_cell[1] == self.finish_line:
                    return 2
                else:
                    return 1
            prev_cell = cc
        return 0

    def make_move(self, state, policy):
        """
        given current state (location and speed), applies given policy, returns new state (location and speed) or the
        fact that the car has crossed the finish line. No need to return reward, it's -1 for every turn
        :param state:
        :param policy:
        :return:
        """
        new_speed = np.zeros(2)
        while new_speed == np.zeros(2) or np.any(new_speed < 0) or np.any(new_speed > 5):
            action = self.sample_index(policy[state])
            new_speed = state[2:] + action - 1
        new_position = state[:2] + new_speed
        crossed_boundary = self.check_boundary_cross(state[:2], new_position)
        if crossed_boundary == 0:
            return action, np.concatenate([new_position, new_speed])
        elif crossed_boundary == 1:
            return action, np.array([np.random.choice(np.arange(3,9)), 31, 0, 0])
        elif crossed_boundary == 2:
            return action, "finished"

    def generate_episode(self, start, policy):
        actions = []
        cur_start = start
        cur_state = np.array(*cur_start, [0, 0])
        states = [cur_state]
        proceed = True
        while proceed:
            action, state = self.make_move(cur_state, policy)
            actions.append(action)
            if state == "finished":
                return actions, states
            states.append(state)
            cur_state = state

    def policy_update_iteration(self):
        states, actions = self.generate_episode(random.choice(np.argwhere(self.field == 1)), self.random_policy)
        g = 0
        w = 1
        for i in range(len(states) - 1, -1, -1):
            g = self.gamma*g - 1
            self.c[*states[i], *actions[i]] += w
            self.q[*states[i], *actions[i]] += w/self.c[*states[i], *actions[i]] * (g - self.q[*states[i], *actions[i]])
            self.policy[*states[i],] = 0
            self.policy[*states[i], *np.argmax(self.q[states[i]])] = 1
            if np.argmax(self.q[states[i]]) != actions:
                return
            w /= 1/9

    def policy_update(self, n_iter=1000):
        for i in range(n_iter):
            self.policy_update_iteration()


if __name__ == "__main__":
    t = RaceTrack()
    t.policy_update()
    print(t.generate_episode([0, 2], t.policy))
