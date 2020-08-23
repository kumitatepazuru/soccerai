import os, signal
import time
from socket import socket, AF_INET, SOCK_DGRAM

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.losses import huber_loss


import args
import shutil
import subprocess
import sys
import threading
import numpy as np

import env_data

args = args.args()


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
                                          (
                                                  env.max_ball_angle - env.min_ball_angle) / env.ball_angle_interval + 1) * int(
                (env.max_goal_distance - env.min_goal_distance) / env.goal_distance_interval + 1) * int(
                (env.max_goal_angle - env.min_goal_angle) / env.goal_angle_interval + 1)))
        else:
            self.V = V
        print("V table:" + readable_size(sys.getsizeof(self.V)))


# 経験メモリの定義
class Memory:
    # 初期化
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    # 経験の追加
    def add(self, experience):
        self.buffer.append(experience)

    # バッチサイズ分の経験をランダムに取得
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    # 経験メモリのサイズ
    def __len__(self):
        return len(self.buffer)


# 行動価値関数の定義
class QNetwork:
    # 初期化
    def __init__(self, state_size, action_size):
        # モデルの作成
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=state_size))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        # モデルのコンパイル
        self.model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))


class ActorCritic:
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.save = 0
        # パラメータの準備
        # 探索パラメータ
        self.E_START = 1.0  # εの初期値
        self.E_STOP = 0.01  # εの最終値
        self.E_DECAY_RATE = 0.00005  # εの減衰率

        # メモリパラメータ
        self.MEMORY_SIZE = 10000  # 経験メモリのサイズ
        self.BATCH_SIZE = 4  # バッチサイズ

    def train(self, name, env, data, logging, gamma, learning_rate, save_interval):
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

    def train2(self, name, env, data, logging, gamma, learning_rate, save_interval, logdir):
        state_size = 4  # 行動数
        action_size = len(env)  # 状態数

        # main-networkの作成
        main_qn = QNetwork(state_size, action_size)

        # target-networkの作成
        target_qn = QNetwork(state_size, action_size)

        # 経験メモリの作成
        memory = Memory(self.MEMORY_SIZE)

        # 学習の開始

        # 環境の初期化
        state = env.reset(name)
        state = np.reshape(state, [1, state_size])

        # エピソード数分のエピソードを繰り返す
        total_step = 0  # 総ステップ数
        episode = 0
        while True:
            step = 0  # ステップ数

            # target-networkの更新
            target_qn.model.set_weights(main_qn.model.get_weights())
            done = False
            episode += 1

            # 1エピソードのループ
            while not done:
                step += 1
                total_step += 1

                # εを減らす
                epsilon = self.E_STOP + (self.E_START - self.E_STOP) * np.exp(-self.E_DECAY_RATE * total_step)

                # ランダムな行動を選択
                if epsilon > np.random.rand():
                    action = env.random()
                # 行動価値関数で行動を選択
                else:
                    action = np.argmax(main_qn.model.predict(state)[0])

                # 行動に応じて状態と報酬を得る
                next_state, done, reward = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                # エピソード完了時
                if done:
                    # 次の状態に状態なしを代入
                    next_state = np.zeros(state.shape)

                    # 経験の追加
                    memory.add((state, action, reward, next_state))
                # エピソード完了でない時
                else:

                    # 経験の追加
                    memory.add((state, action, reward, next_state))

                    # 状態に次の状態を代入
                    state = next_state

                # 行動価値関数の更新
                if len(memory) >= self.BATCH_SIZE:
                    # ニューラルネットワークの入力と出力の準備
                    inputs = np.zeros((self.BATCH_SIZE, 4))  # 入力(状態)
                    targets = np.zeros((self.BATCH_SIZE, 6))  # 出力(行動ごとの価値)

                    # バッチサイズ分の経験をランダムに取得
                    minibatch = memory.sample(self.BATCH_SIZE)

                    # ニューラルネットワークの入力と出力の生成
                    for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):

                        # 入力に状態を指定
                        inputs[i] = state_b

                        # 採った行動の価値を計算
                        if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                            target = reward_b + gamma * np.amax(
                                target_qn.model.predict(next_state_b)[0])
                        else:
                            target = reward_b

                        # 出力に行動ごとの価値を指定
                        targets[i] = main_qn.model.predict(state_b)
                        targets[i][action_b] = target  # 採った行動の価値

                    # 行動価値関数の更新
                    main_qn.model.fit(inputs, targets, epochs=1, verbose=0)

            # エピソード完了時のログ表示
            print('エピソード: {}, ステップ数: {}, epsilon: {:.4f}'.format(episode, step, epsilon))


def train(arguments, ip=0):
    # スレッド化して実行
    trainer = ActorCritic(Actor, Critic)
    num = arguments.player_num_1
    if arguments.file1 is not None:
        tmp = np.load(arguments.file1)
    else:
        tmp = None
    for i in range(num):
        env = env_data.env(arguments.address, arguments.host,
                           arguments.server_port * (arguments.auto_ip * -1 + 1) + ip * arguments.auto_ip,
                           arguments.send_log, arguments.recieve_log, arguments.analysis_log, arguments.logdir,
                           arguments.max_ball_distance, arguments.min_ball_distance, arguments.ball_distance_interval,
                           arguments.special_division, arguments.special_division_value,
                           arguments.damage_reset_ball_distance,
                           arguments.max_ball_angle, arguments.min_ball_angle, arguments.ball_angle_interval,
                           arguments.max_goal_distance, arguments.min_goal_distance, arguments.goal_distance_interval,
                           arguments.max_goal_angle, arguments.min_goal_angle, arguments.goal_angle_interval,
                           num, arguments.noise, arguments.actions, arguments.goal_reset)
        th1 = threading.Thread(target=trainer.train2, args=[
            arguments.team_name_1,
            env,
            tmp,
            arguments.logging,
            arguments.gamma,
            arguments.learning_rate,
            arguments.save_interval,
            arguments.logdir
        ])
        th1.setDaemon(True)
        th1.start()

    if not arguments.only_team_1:
        trainer = ActorCritic(Actor, Critic)
        num = arguments.player_num_2
        if arguments.file2 is not None:
            tmp = np.load(arguments.file2)
        else:
            tmp = None
        for i in range(num):
            env = env_data.env(arguments.address, arguments.host,
                               arguments.server_port * (arguments.auto_ip * -1 + 1) + ip * arguments.auto_ip,
                               arguments.send_log, arguments.recieve_log, arguments.analysis_log, arguments.logdir,
                               arguments.max_ball_distance, arguments.min_ball_distance,
                               arguments.ball_distance_interval,
                               arguments.special_division,
                               arguments.special_division_value, arguments.damage_reset_ball_distance,
                               arguments.max_ball_angle, arguments.min_ball_angle, arguments.ball_angle_interval,
                               arguments.max_goal_distance, arguments.min_goal_distance,
                               arguments.goal_distance_interval,
                               arguments.max_goal_angle, arguments.min_goal_angle, arguments.goal_angle_interval,
                               num, arguments.noise, arguments.actions, arguments.goal_reset)
            th2 = threading.Thread(target=trainer.train2, args=[
                arguments.team_name_2,
                env,
                tmp,
                arguments.logging,
                arguments.gamma,
                arguments.learning_rate,
                arguments.save_interval,
                arguments.logdir
            ])
            th2.setDaemon(True)
            th2.start()
    return th1


if __name__ == "__main__":
    try:
        shutil.rmtree("./" + args.logdir)
    except FileNotFoundError:
        pass
    s = socket(AF_INET, SOCK_DGRAM)
    s.bind((args.host, 0))
    p = s.getsockname()[1]
    s.close()
    tensorboard = subprocess.Popen(["tensorboard", "--logdir", "./" + args.logdir, "--port", str(p)])
    if args.auto_server:
        cmd = ["rcssserver",
               "server::game_logging=false", "server::text_logging=false", "server::nr_normal_halfs=1",
               "server::use_offside=off",
               "server::synch_mode=true", "server::free_kick_faults=false", "server::forbid_kick_off_offside=false",
               "server::penalty_shoot_outs=false", "server::half_time=300", "server::nr_normal_halfs=9999",
               "server::nr_extra_halfs=0", "server::stamina_capacity=-1", "server::drop_ball_time=200",
               "server::auto_mode=true",
               "server::connect_wait=25", "server::kick_off_wait=10", "server::game_over_wait=10"]
        if args.auto_ip:
            print("\033[38;5;12m[INFO]\t\033[38;5;13mSearching for available ports ...\033[0m")
            s = socket(AF_INET, SOCK_DGRAM)
            s.bind((args.host, 0))
            p1 = s.getsockname()[1]
            s.close()
            s = socket(AF_INET, SOCK_DGRAM)
            s.bind((args.host, 0))
            p2 = s.getsockname()[1]
            s.close()
            s = socket(AF_INET, SOCK_DGRAM)
            s.bind((args.host, 0))
            p3 = s.getsockname()[1]
            s.close()
            print("\033[38;5;10m[OK] server port:", p1, p2, p3, "\033[0m")

            server = subprocess.Popen(
                cmd + ["server::port=" + str(p1),
                       "server::coach_port=" + str(p2), "server::olcoach_port=" + str(p3)])
        else:
            server = subprocess.Popen(
                cmd + ["server::port=" + str(args.server_port),
                       "server::coach_port=" + str(args.server_port + 1),
                       "server::olcoach_port=" + str(args.server_port + 2)])

    if args.auto_window:
        if args.auto_ip:
            window = subprocess.Popen(["soccerwindow2", "--port " + str(p1)])
        else:
            window = subprocess.Popen(["soccerwindow2", "--port " + str(args.server_port)])
    if args.auto_browser:
        browser = subprocess.Popen(["firefox", os.uname()[1] + ":" + str(p)])
    if args.auto_ip:
        th = train(args, p1)
    else:
        th = train(args)
    if args.auto_server:
        try:
            server.wait()
        except KeyboardInterrupt:
            if args.auto_window:
                os.killpg(os.getpgid(window.pid), signal.SIGQUIT)
            os.killpg(os.getpgid(tensorboard.pid), signal.SIGQUIT)
            sys.exit()
        if server.returncode != 0:
            print(
                "\033[38;5;1m[ERR]\033[0m\033[38;5;12mThe soccer server has been shut down for some reason. "
                "Please check if the following ports are free.\n\033[0m",
                "\033[4m" + str(args.server_port) + "\033[0m", ",", "\033[4m" + str(args.server_port + 1) + "\033[0m",
                ",", "\033[4m" + str(args.server_port + 2) + "\033[0m")
            if args.auto_window:
                os.killpg(os.getpgid(window.pid), signal.SIGQUIT)
            os.killpg(os.getpgid(tensorboard.pid), signal.SIGQUIT)
            sys.exit()
    try:
        th.join()
    except KeyboardInterrupt:
        if args.auto_window:
            os.killpg(os.getpgid(window.pid), signal.SIGQUIT)
        os.killpg(os.getpgid(tensorboard.pid), signal.SIGQUIT)
        sys.exit()
