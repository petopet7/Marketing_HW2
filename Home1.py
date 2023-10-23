import random
import matplotlib.pyplot as plt
import csv
import numpy as np
from abc import ABC, abstractmethod


class Bandit:
    @abstractmethod
    def __init__(self, p):
        self.true_means = p
        self.estimated_means = [0.0] * len(p)
        self.action_counts = [0] * len(p)
        self.cumulative_rewards = [0.0]

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = self.true_means[arm]
            self.update(arm, reward)
            rewards.append(reward)
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)
        return rewards

    def report(self, rewards, algorithm_name):
        self.save_rewards_to_csv(rewards, algorithm_name)
        self.plot_learning_process(rewards, algorithm_name)
        self.print_cumulative_metrics(rewards, algorithm_name)

    def save_rewards_to_csv(self, rewards, algorithm_name):
        with open('bandit_rewards.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for reward in rewards:
                csvwriter.writerow([algorithm_name, reward])

    def plot_learning_process(self, rewards, algorithm_name):
        plt.plot(self.cumulative_rewards[1:], label=algorithm_name)
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Rewards')
        plt.legend()

    def print_cumulative_metrics(self, rewards, algorithm_name):
        cumulative_reward = sum(rewards)
        cumulative_regret = max(self.true_means) * len(rewards) - cumulative_reward
        print(f"{algorithm_name} Cumulative Reward: {cumulative_reward}")
        print(f"{algorithm_name} Cumulative Regret: {cumulative_regret}")

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon):
        super().__init__(p)
        self.epsilon = epsilon

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.true_means) - 1)  # Explore by choosing a random arm
        else:
            return self.estimated_means.index(max(self.estimated_means))  # Exploit by choosing the arm with the highest estimated mean

    def update(self, arm, reward):
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.estimated_means[arm] += (1 / n) * (reward - self.estimated_means[arm])  # Update the estimated mean using the sample average method
        self.epsilon *= 0.99  # Decay epsilon by multiplying it with a factor less than 1


class ThompsonSampling(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.alpha = [1] * len(p)
        self.beta = [1] * len(p)

    def pull(self):
        sampled_means = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.true_means))]
        return sampled_means.index(max(sampled_means))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


class Visualization():

    def plot1(self, eg_rewards, ts_rewards):
        # Calculate cumulative average rewards for E-Greedy and Thompson Sampling
        eg_cumulative_rewards = np.cumsum(eg_rewards)
        ts_cumulative_rewards = np.cumsum(ts_rewards)

        eg_avg_rewards = eg_cumulative_rewards / (np.arange(len(eg_rewards)) + 1)
        ts_avg_rewards = ts_cumulative_rewards / (np.arange(len(ts_rewards)) + 1)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(eg_avg_rewards, label="Epsilon-Greedy")
        plt.plot(ts_avg_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.title("Learning Process of E-Greedy and Thompson Sampling")
        plt.legend()
        plt.grid(True)

    def plot2(self, eg_rewards, ts_rewards):
        # Calculate cumulative rewards for E-Greedy and Thompson Sampling
        eg_cumulative_rewards = np.cumsum(eg_rewards)
        ts_cumulative_rewards = np.cumsum(ts_rewards)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(eg_cumulative_rewards, label="Epsilon-Greedy")
        plt.plot(ts_cumulative_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards of E-Greedy and Thompson Sampling")
        plt.legend()
        plt.grid(True)

# Usage
Bandit_Reward = [0.1, 0.3, 0.5, 0.7]
NumberOfTrials = 20000

epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, 0.1)
epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(NumberOfTrials)
epsilon_greedy_bandit.report(epsilon_greedy_rewards, 'Epsilon Greedy')

thompson_bandit = ThompsonSampling(Bandit_Reward)
thompson_rewards = thompson_bandit.experiment(NumberOfTrials)
thompson_bandit.report(thompson_rewards, 'Thompson Sampling')

# Create an instance of the Visualization class
visualization = Visualization()
visualization.plot1(epsilon_greedy_rewards, thompson_rewards)
visualization.plot2(epsilon_greedy_rewards, thompson_rewards)

plt.show()
