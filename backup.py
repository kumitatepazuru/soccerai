import threading
import time

import numpy as np

import env_data


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Actor:

    def __init__(self, env):
        self.actions = list(range(len(env)))
        self.Q = np.random.uniform(0, 1, (
            ((env.max_ball_distance - env.min_ball_distance) * 10) + 1,
            ((env.max_ball_angle - env.min_ball_angle) * 2 * 10) + 1, len(env)))

    def policy(self, ball_distance, ball_angle):
        a = np.random.choice(self.actions, 1,
                             p=softmax(self.Q[ball_distance][ball_angle]))
        return a[0]


class Critic:

    def __init__(self, env):
        self.V = np.zeros((((env.max_ball_distance - env.min_ball_distance) * 10) + 1) * (
                ((env.max_ball_angle - env.min_ball_angle) * 10) + 1) * len(env))


class ActorCritic:
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, name,env, gamma=0.9, learning_rate=0.1):
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        s = env.reset(name)
        done = False
        while not done:
            a = actor.policy(s[0],s[1])
            n_state, reward, done = env.step(a)

            gain = reward + gamma * critic.V[n_state[0] * n_state[1]]
            estimated = critic.V[s[0] * s[1]]
            td = gain - estimated
            actor.Q[s[0]][s[1]][a] += learning_rate * td
            critic.V[s[0] * s[1]] += learning_rate * td
            s = n_state

        return actor, critic


def train():
    time.sleep(1)
    trainer = ActorCritic(Actor, Critic)
    for i in range(11):
        env = env_data.env()
        threading.Thread(target=trainer.train, args=["AI1", env]).start()
    trainer = ActorCritic(Actor, Critic)
    for i in range(11):
        env = env_data.env()
        threading.Thread(target=trainer.train, args=["AI2", env]).start()
    env.show_log()


if __name__ == "__main__":
    train()
