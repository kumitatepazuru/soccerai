import concurrent.futures
import random
import sys
import threading
import time

import numpy as np

import env_data


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def readable_size(size):
    for unit in ['K', 'M']:
        if abs(size) < 1024.0:
            return "%.1f%sB" % (size, unit)
        size /= 1024.0
    size /= 1024.0
    return "%.1f%s" % (size, 'GB')


class Actor:

    def __init__(self, env, Q):
        self.actions = list(range(len(env)))
        # ballの距離・ballの角度・actionの種類すべてが格納されていて、対応するindexがActor.policyで選ばれる。
        if Q is None:
            self.Q = np.random.uniform(0, 1, (
                int((env.max_ball_distance - env.min_ball_distance) / env.ball_distance_interval + 1),
                int((env.max_ball_angle - env.min_ball_angle) / env.ball_angle_interval + 1), len(env)))
        else:
            self.Q = Q
        print("Q table:" + readable_size(sys.getsizeof(self.Q)))

    def policy(self, ball_distance, ball_angle):
        a = np.random.choice(self.actions, 1,
                             p=softmax(self.Q[ball_distance][ball_angle]))
        return a[0]


class Critic:

    def __init__(self, env, V):
        # self.Qとほぼ同じ
        if V is None:
            self.V = np.zeros(int(((env.max_ball_distance - env.min_ball_distance) / env.ball_distance_interval + 1) * (
                    (env.max_ball_angle - env.min_ball_angle) / env.ball_angle_interval + 1) * len(env)))
        else:
            self.V = V
        print("V table:" + readable_size(sys.getsizeof(self.V)))


class ActorCritic:
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.save = 0

    def train(self, name, env, data=None,gamma=0.9, learning_rate=0.1, save_interval=500):
        # Actor Critic法の実装
        if data is not None:
            print(gamma)
            actor = self.actor_class(env,data["arr_0"])
            critic = self.critic_class(env,data["arr_1"])
        else:
            actor = self.actor_class(env, None)
            critic = self.critic_class(env, None)
        s = env.reset(name)
        i = 0
        while True:
            if i % save_interval == 0 and self.save == 0:
                self.save = 1
                np.savez("agent_data_" + name + "_" + str(i), actor.Q, critic.V)
            elif i % int(save_interval/2) == 0 and self.save == 1:
                self.save = 0
            a = actor.policy(s[0], s[1])
            n_state, reward, end_action = env.step(a)

            gain = reward + gamma * critic.V[n_state[0] * n_state[1]]
            estimated = critic.V[s[0] * s[1]]
            td = gain - estimated
            actor.Q[s[0]][s[1]][end_action] += learning_rate * td
            if actor.Q[s[0]][s[1]][end_action] > 100:
                print("WARNING: Q table exceeds the threshold value.")
                actor.Q[s[0]][s[1]][end_action] /= 100
            critic.V[s[0] * s[1]] += learning_rate * td
            s = n_state
            i += 1


def train():
    # スレッド化して実行
    trainer = ActorCritic(Actor, Critic)
    if len(sys.argv) == 3:
        tmp = np.load(sys.argv[1])
        for i in range(11):
            env = env_data.env()
            threading.Thread(target=trainer.train, args=["AI1", env,tmp]).start()
    else:
        for i in range(11):
            env = env_data.env()
            threading.Thread(target=trainer.train, args=["AI1", env]).start()
    time.sleep(1)
    trainer = ActorCritic(Actor, Critic)
    if len(sys.argv) == 3:
        tmp = np.load(sys.argv[2])
        for i in range(11):
            env = env_data.env()
            threading.Thread(target=trainer.train, args=["AI2", env, tmp]).start()
    else:
        for i in range(11):
            env = env_data.env()
            threading.Thread(target=trainer.train, args=["AI2", env]).start()
    env.show_log()


if __name__ == "__main__":
    train()
