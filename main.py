import os, signal
from socket import socket, AF_INET, SOCK_DGRAM
import args
import shutil
import subprocess
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
                                          (
                                                  env.max_ball_angle - env.min_ball_angle) / env.ball_angle_interval + 1) * int(
                (env.max_goal_distance - env.min_goal_distance) / env.goal_distance_interval + 1) * int(
                (env.max_goal_angle - env.min_goal_angle) / env.goal_angle_interval + 1)))
        else:
            self.V = V
        print("V table:" + readable_size(sys.getsizeof(self.V)))


class ActorCritic:
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.save = 0

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
        th1 = threading.Thread(target=trainer.train, args=[
            arguments.team_name_1,
            env,
            tmp,
            arguments.logging,
            arguments.gamma,
            arguments.learning_rate,
            arguments.save_interval
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
            th2 = threading.Thread(target=trainer.train, args=[
                arguments.team_name_2,
                env,
                tmp,
                arguments.logging,
                arguments.gamma,
                arguments.learning_rate,
                arguments.save_interval
            ])
            th2.setDaemon(True)
            th2.start()
    return th1


if __name__ == "__main__":
    args = args.args()
    try:
        shutil.rmtree("./" + args.logdir)
    except FileNotFoundError:
        pass
    tensorboard = subprocess.Popen(["tensorboard", "--logdir", "./" + args.logdir, "--port", "0"])
    if args.auto_server:
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
            print("\033[38;5;10m[OK] server port:", p1, p2, p3)

            server = subprocess.Popen(
                ["rcssserver",
                 "server::game_logging=false", "server::text_logging=false", "server::nr_normal_halfs=1",
                 "server::use_offside=off",
                 "server::synch_mode=true", "server::free_kick_faults=false", "server::forbid_kick_off_offside=false",
                 "server::penalty_shoot_outs=false", "server::half_time=-1",
                 "server::stamina_capacity=-1",
                 "server::drop_ball_time=500", "server::port=" + str(p1),
                 "server::coach_port=" + str(p2), "server::olcoach_port=" + str(p3), "server::auto_mode=true",
                 "server::connect_wait=25", "server::kick_off_wait=10", "server::game_over_wait=10"])
        else:
            server = subprocess.Popen(
                ["rcssserver",
                 "server::game_logging=false", "server::text_logging=false", "server::nr_normal_halfs=1",
                 "server::use_offside=off",
                 "server::synch_mode=true", "server::free_kick_faults=false", "server::forbid_kick_off_offside=false",
                 "server::penalty_shoot_outs=false", "server::half_time=-1",
                 "server::stamina_capacity=-1",
                 "server::drop_ball_time=500", "server::port=" + str(args.server_port),
                 "server::coach_port=" + str(args.server_port + 1), "server::olcoach_port=" + str(args.server_port + 2),
                 "server::auto_mode=true", "server::connect_wait=25", "server::kick_off_wait=10",
                 "server::game_over_wait=10"])

    if args.auto_window:
        if args.auto_ip:
            window = subprocess.Popen(["soccerwindow2", "--port " + str(p1)])
        else:
            window = subprocess.Popen(["soccerwindow2", "--port " + str(args.server_port)])
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
