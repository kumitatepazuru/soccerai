import sys
import threading

import numpy as np

import env_data


def softmax(x):
    # 入力値の中で最大値を取得
    c = np.max(x)
    # オーバーフロー対策として、最大値cを引く。こうすることで値が小さくなる
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a, axis=0)

    y = exp_a / sum_exp_a
    return y


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
                int((env.max_ball_distance - env.min_ball_distance) / env.ball_distance_interval + 1 + env.
                    special_division),
                int((env.max_ball_angle - env.min_ball_angle) / env.ball_angle_interval + 1),
                int((env.max_goal_distance - env.min_goal_distance) / env.goal_distance_interval + 1),
                int((env.max_goal_angle - env.min_goal_angle) / env.goal_angle_interval + 1),
                len(env)))
        else:
            self.Q = Q
        print("Q table:" + readable_size(sys.getsizeof(self.Q)))

    def policy(self, ball_distance, ball_angle, goal_distance, goal_angle):
        a = np.random.choice(self.actions, 1,
                             p=softmax(self.Q[ball_distance][ball_angle][goal_distance][goal_angle]))
        return a[0]


class Critic:

    def __init__(self, env, V):
        # self.Qとほぼ同じ
        if V is None:
            self.V = np.zeros(int(((env.max_ball_distance - env.min_ball_distance) / env.ball_distance_interval + 1 +
                                   env.special_division) * (
                    (env.max_ball_angle - env.min_ball_angle) / env.ball_angle_interval + 1) * int(
                (env.max_goal_distance - env.min_goal_distance) / env.goal_distance_interval + 1) * int(
                (env.max_goal_angle - env.min_goal_angle) / env.goal_angle_interval + 1) * len(env)))
        else:
            self.V = V
        print("V table:" + readable_size(sys.getsizeof(self.V)))


class ActorCritic:
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.save = 0

    def train(self, name, env, data=None, logging=False, gamma=0.95, learning_rate=0.1, save_interval=25000):
        # Actor Critic法の実装
        if data is not None:
            actor = self.actor_class(env, data["arr_0"])
            critic = self.critic_class(env, data["arr_1"])
        else:
            actor = self.actor_class(env, None)
            critic = self.critic_class(env, None)
        s = env.reset(name)
        i = 0
        while True:
            if i % save_interval == 0 and self.save == 0 and logging:
                self.save = 1
                np.savez("agent_data_" + name + "_" + str(i), actor.Q, critic.V)
            elif i % int(save_interval / 2) == 0 and self.save == 1:
                self.save = 0
            a = actor.policy(s[0], s[1], s[2], s[3])
            n_state, reward, end_action = env.step(a)

            gain = reward + gamma * critic.V[n_state[0] * n_state[1] * n_state[2] * n_state[3]]
            estimated = critic.V[s[0] * s[1] * s[2] * s[3]]
            td = gain - estimated
            actor.Q[s[0]][s[1]][s[2]][s[3]][end_action] += learning_rate * td
            critic.V[s[0] * s[1] * s[2] * s[3]] += learning_rate * td
            s = n_state
            i += 1


def train():
    # スレッド化して実行
    trainer = ActorCritic(Actor, Critic)
    num = 1
    if len(sys.argv) == 3:
        tmp = np.load(sys.argv[1])
        for i in range(num):
            env = env_data.env(num=num)
            threading.Thread(target=trainer.train, args=["AI1", env, tmp]).start()
    else:
        for i in range(num):
            env = env_data.env(num=num)
            threading.Thread(target=trainer.train, args=["AI1", env]).start()

    # if len(sys.argv) == 3:
    #     tmp = np.load(sys.argv[2])
    #     for i in range(num):
    #         env = env_data.env(num=num)
    #         threading.Thread(target=trainer.train, args=["AI2", env, tmp]).start()
    # else:
    #     for i in range(num):
    #         env = env_data.env(num=num)
    #         threading.Thread(target=trainer.train, args=["AI2", env]).start()


if __name__ == "__main__":
    train()
