import numpy as np
from scipy.stats import poisson

LAMBDA_ONE_REQUEST = 3
LAMBDA_ONE_RETURN = 3
LAMBDA_TWO_REQUEST = 4
LAMBDA_TWO_RETURN = 2
GAMMA = 0.9
MAX_CARS = 20
MAX_MOVED_CARS = 5
RENTAL_REWARD = 10
MOVED_FEE = 2


# states are number of cars at both places at the end of the day, actions are the amount of cars moved from first to
# second place (second to first are negative numbers)


class JCR:
    def __init__(self, is_parking_fee=False, free_one_to_two=False):
        self.values = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
        self.req_probs_one = poisson(LAMBDA_ONE_REQUEST).pmf(np.arange(MAX_CARS + 1))
        self.req_probs_one[-1] += 1 - self.req_probs_one.sum()
        self.ret_probs_one = poisson(LAMBDA_ONE_RETURN).pmf(np.arange(MAX_CARS + 1))
        self.ret_probs_one[-1] += 1 - self.ret_probs_one.sum()
        self.req_probs_two = poisson(LAMBDA_TWO_REQUEST).pmf(np.arange(MAX_CARS + 1))
        self.req_probs_two[-1] += 1 - self.req_probs_two.sum()
        self.ret_probs_two = poisson(LAMBDA_TWO_RETURN).pmf(np.arange(MAX_CARS + 1))
        self.ret_probs_two[-1] += 1 - self.ret_probs_two.sum()
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
        self.reward_cars_left_probs_one = self.fill_reward_cars_left_probs(self.req_probs_one, self.ret_probs_one)
        self.reward_cars_left_probs_two = self.fill_reward_cars_left_probs(self.req_probs_two, self.ret_probs_two)
        self.is_parking_fee = is_parking_fee
        self.free_one_to_two = free_one_to_two

    @staticmethod
    def _cars_left(cars_before, taken_prob, returned_prob):
        """
        :param cars_before:
        :param taken_prob:
        :param returned_prob:
        :return: 2d array, first dimension: cars left, second dimension - rewards/10, values: probability
        """
        result = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        result[np.arange(cars_before, -1, -1), np.arange(cars_before + 1)] = taken_prob[:cars_before + 1]
        result[0, cars_before] += taken_prob[cars_before + 1:].sum()
        for j in range(cars_before + 1):
            returned_probs_adjusted = returned_prob[:MAX_CARS - cars_before + 1 + j].copy()
            returned_probs_adjusted[-1] += returned_prob[MAX_CARS - cars_before + 1 + j:].sum()
            result[cars_before - j:, j] = returned_probs_adjusted * result[cars_before - j, j]
        # print(result.sum())
        return result

    def fill_reward_cars_left_probs(self, taken_prob, returned_prob):
        probs = np.zeros((MAX_CARS + 1, MAX_CARS + 1, MAX_CARS + 1))
        for k in range(MAX_CARS + 1):
            probs[k, :, :] = self._cars_left(k, taken_prob, returned_prob)
        return probs

    def evaluate_rewards_elementwise(self, i, j, action, is_parking_fee=False, free_one_to_two=False):
        """
        computes expected reward after taking action `action` from state [i, j]
        :param i: amount of cars in first location
        :param j: amount of cars in second location
        :param action: how many cars to move from first to second location
        :param is_parking_fee: does Jack pay 4$ if >10 cars are spending the night on location
        :param free_one_to_two: can one car be moved for free from first to second location
        :return:
        """
        cars_to_move = max(min(i, action), -j)
        a = min(i - cars_to_move, MAX_CARS)
        b = min(j + cars_to_move, MAX_CARS)
        parking_fee = 0
        if is_parking_fee:
            if a > 10:
                parking_fee += 4
            if b > 10:
                parking_fee += 4
        first_probs = self.reward_cars_left_probs_one[a]
        second_probs = self.reward_cars_left_probs_two[b]
        exp_reward_one = (first_probs.sum(axis=0) * np.arange(0, (MAX_CARS + 1) * RENTAL_REWARD, RENTAL_REWARD)).sum()
        exp_reward_two = (second_probs.sum(axis=0) * np.arange(0, (MAX_CARS + 1) * RENTAL_REWARD, RENTAL_REWARD)).sum()
        moving_cars_fee = MOVED_FEE * (abs(cars_to_move))
        if free_one_to_two:
            if cars_to_move > 0:
                moving_cars_fee -= MOVED_FEE
        exp_reward = exp_reward_one + exp_reward_two - moving_cars_fee - parking_fee
        exp_state = ((first_probs.sum(axis=1).reshape(-1, 1).dot(
            second_probs.sum(axis=1).reshape(1, -1))) * GAMMA * self.values).sum()
        return exp_reward + exp_state

    def evaluate_policy(self, theta=0.01):
        delta = 1
        while delta >= theta:
            delta = 0
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    v = self.values[i, j]
                    self.values[i, j] = self.evaluate_rewards_elementwise(
                        i, j, self.policy[i, j], is_parking_fee=self.is_parking_fee, free_one_to_two=self.free_one_to_two)
                    delta = max(delta, abs(v - self.values[i, j]))

    def improve_policy(self):
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = self.policy[i, j]
                action_values = []
                for cars_to_move in range(max(-j, i - MAX_CARS, -MAX_MOVED_CARS),
                                          min(i, MAX_CARS - j, MAX_MOVED_CARS) + 1):
                    action_values.append(self.evaluate_rewards_elementwise(
                        i, j, cars_to_move, is_parking_fee=self.is_parking_fee, free_one_to_two=self.free_one_to_two))
                best_action = np.argmax(action_values) + max(-j, i - MAX_CARS, -MAX_MOVED_CARS)
                self.policy[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False
        return policy_stable

    def find_best_policy(self, theta=0.01):
        counter = 0
        while True:
            self.evaluate_policy(theta=theta)
            done = self.improve_policy()
            counter += 1
            if done:
                print(counter)
                return


if __name__ == "__main__":
    jcr = JCR(is_parking_fee=False, free_one_to_two=False)
    jcr.find_best_policy()
