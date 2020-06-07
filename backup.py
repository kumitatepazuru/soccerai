import sys
import threading

import numpy as np

import env_data


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def readable_size(size):
    for unit in ['K','M']:
        if abs(size) < 1024.0:
            return "%.1f%sB" % (size,unit)
        size /= 1024.0
    size /= 1024.0
    return "%.1f%s" % (size,'GB')


class Actor:

    def __init__(self, env):
        self.actions = list(range(len(env)))
        # ballの距離・ballの角度・actionの種類すべてが格納されていて、対応するindexがActor.policyで選ばれる。
        self.Q = np.random.uniform(0, 1, (
            (env.max_ball_distance - env.min_ball_distance) + 1,
            (env.max_ball_angle - env.min_ball_angle) + 1, len(env)))
        print("Q table:"+readable_size(sys.getsizeof(self.Q)))

    def policy(self, ball_distance, ball_angle):
        a = np.random.choice(self.actions, 1,
                             p=softmax(self.Q[ball_distance][ball_angle]))
        return a[0]


class Critic:

    def __init__(self, env):
        # self.Qとほぼ同じ
        self.V = np.zeros(((env.max_ball_distance - env.min_ball_distance) + 1) * (
                (env.max_ball_angle - env.min_ball_angle) + 1) * len(env))
        print("V table:" + readable_size(sys.getsizeof(self.V)))


class ActorCritic:
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, name, env, gamma=0.9, learning_rate=0.1):
        # Actor Critic法の実装
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        s = env.reset(name)
        done = False
        while not done:
            a = actor.policy(s[0], s[1])
            n_state, reward, done = env.step(a)

            gain = reward + gamma * critic.V[n_state[0] * n_state[1]]
            estimated = critic.V[s[0] * s[1]]
            td = gain - estimated
            actor.Q[s[0]][s[1]][a] += learning_rate * td
            critic.V[s[0] * s[1]] += learning_rate * td
            s = n_state

        return actor, critic


def train():
    # スレッド化して実行
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
