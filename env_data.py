import random

import libscips.player
import numpy as np
import tensorflow.compat.v1 as tf


def logtoplot(data, interval):
    return list(map(lambda n: n.mean(), np.array_split(np.array(data), interval)))


class TensorBoardLogger(object):

    def __init__(self, log_dir, session=None):
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.episode = 0

        self.session = session or tf.get_default_session() or tf.Session()

    def log(self, logs=None):
        # scalar logging
        if logs is None:
            logs = {}
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, self.episode)

        self.writer.flush()
        self.episode += 1


class env(libscips.player.player_signal):
    def __init__(self, ADDRESS, HOST, SERVER_PORT, send_log, recieve_log, analysis_log, logdir,
                 max_ball_distance, min_ball_distance, ball_distance_interval, special_division,
                 special_division_value, damage_reset_ball_distance,
                 max_ball_angle, min_ball_angle, ball_angle_interval,
                 max_goal_distance, min_goal_distance, goal_distance_interval,
                 max_goal_angle, min_goal_angle, goal_angle_interval,
                 num, noise, actions, goal_reset):
        # 初期設定
        super().__init__(ADDRESS, HOST, SERVER_PORT, send_log, recieve_log, analysis_log)
        self.logger1 = TensorBoardLogger(log_dir='./'+logdir+'/reward')
        self.logger2 = TensorBoardLogger(log_dir='./'+logdir+'/see')
        self.logger3 = TensorBoardLogger(log_dir='./'+logdir+'/action')
        self.logger4 = TensorBoardLogger(log_dir='./'+logdir+'/damage')
        self.logger5 = TensorBoardLogger(log_dir='./'+logdir+'/goal')

        self.max_ball_distance = max_ball_distance
        self.min_ball_distance = min_ball_distance
        self.ball_distance_interval = ball_distance_interval
        self.max_ball_angle = max_ball_angle
        self.min_ball_angle = min_ball_angle
        self.ball_angle_interval = ball_angle_interval
        self.max_goal_distance = max_goal_distance
        self.min_goal_distance = min_goal_distance
        self.goal_distance_interval = goal_distance_interval
        self.max_goal_angle = max_goal_angle
        self.min_goal_angle = min_goal_angle
        self.goal_angle_interval = goal_angle_interval
        self.special_division = special_division
        self.special_division_value = special_division_value
        self.damage_reset_ball_distance = damage_reset_ball_distance
        self.noise = noise
        self.noise_num = 0
        self.ball_distance = 0
        self.goal_distance = 0
        self.actions = actions
        self.kickcount = 0
        self.ball_damage = 0
        self.num = num
        self.team = None
        self.enemy = None
        self.time = 0
        self.goal = 0
        self.reset_kick = self.kickcount
        self.turn_repeat = [0, 0]
        self.goal_reset = goal_reset

    def __len__(self):
        # len(env)をやったときに返すデータ
        return self.actions

    def _update(self, *args):
        """グラフを更新するための関数"""
        self.logger1.log(logs=dict(reward=args[0]))
        self.logger2.log(logs=dict(reward=args[1]))
        self.logger3.log(logs=dict(reward=args[2]))
        self.logger4.log(logs=dict(reward=args[3]))
        self.logger5.log(logs=dict(goal_count=args[4]))

    def reset(self, name):
        # 環境のリセット
        self.team = self.send_init(name)[0][1]
        self.enemy = "l" * (self.team == "r") + "r" * (self.team == "l")
        self.send_move(-10, -10)
        rec = {"type": None}
        rec_raw = []
        while rec["type"] != "hear":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)

        while rec["type"] != "see":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)
        ball_info = self.see_analysis(rec_raw, "b")
        goal_info = self.see_analysis(rec_raw, ["g", self.enemy])

        if ball_info is None:
            ball_info = (self.max_ball_distance + 1, self.max_ball_angle + 1)
        elif float(ball_info[0]) < self.special_division_value and self.special_division:
            ball_info = [self.max_ball_distance + 2, float(ball_info[1])]
        if goal_info is None:
            goal_info = [self.max_goal_distance + 1, self.max_goal_angle + 1]
        if float(goal_info[0]) > self.max_goal_distance:
            goal_info = [self.max_goal_distance, goal_info[1]]
        return int(float(ball_info[0]) / self.ball_distance_interval), int(
            float(ball_info[1]) / self.ball_angle_interval), int(
            float(goal_info[0]) / self.goal_distance_interval), int(float(goal_info[1]) / self.goal_angle_interval)

    def repeat(self, num, out=10):
        if self.turn_repeat[1] != num:
            self.turn_repeat[0] = 0
            self.turn_repeat[1] = num
            return 0
        else:
            self.turn_repeat[0] += 1
            if self.turn_repeat[0] > out:
                return 100
            else:
                return 0

    def step(self, action):
        self.time += 1
        # actionから移動をして報酬を計算するプログラム
        # self.goal_damage += 0.01
        reward = 0
        reward_see = 0
        reward_action = 0
        self.ball_damage += (12 - self.num) * 0.1
        if random.random() < self.noise:
            end_action = action
        else:
            action = random.randrange(0, self.actions)
            end_action = action
        if action == 0:
            self.send_dash(100)
            reward_action -= self.repeat(1, out=20)
        elif action == 1:
            self.send_turn(10)
            reward_action -= self.repeat(1)
        elif action == 2:
            self.send_turn(-10)
            reward_action -= self.repeat(1)
        elif action == 3:
            self.send_turn(30)
            reward_action -= self.repeat(1)
        elif action == 4:
            self.send_turn(-30)
            reward_action -= self.repeat(1)
        elif action == 5:
            self.send_kick(100, 0)

        """==========reward setting=========="""
        # 見えているものを確認する
        rec = {"type": None}
        rec_raw = []

        while rec["type"] != "see":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)
            if rec["type"] == "hear":
                # ゴールしたときの報酬（できたらめっちゃ褒める）
                if rec["contents"][:6] == "goal_" + self.team:
                    self.send_move(-10, -10)
                    self.ball_damage = -1000 * int(rec["contents"][7:])
                    reward_action += 1000 * int(rec["contents"][7:])
                    if int(rec["contents"][7:]) == 1:
                        self.noise_num = 2
                    elif int(rec["contents"][7:]) == 10:
                        self.noise_num = 3
                    self.goal += 1
                    print("GOAL!", int(rec["contents"][7:]), "time:", rec["time"])
            elif rec["type"] == "sense_body":
                if self.kickcount < int(rec["value"][4][1]):
                    self.ball_damage -= (1000 + self.ball_damage) / 5
                    reward_action += 100
                    print("kick!!!!", reward_action, "time:", rec["time"])
                    if self.kickcount == 0:
                        self.noise_num = 1
                    self.kickcount = int(rec["value"][4][1])

        ball_info = self.see_analysis(rec_raw, "b")
        goal_info = self.see_analysis(rec_raw, ["g", self.enemy])

        # ゴールが見えているか
        if goal_info is None:
            goal_info = [self.max_goal_distance + 1, self.max_goal_angle + 1]
        else:
            goal_info = list(map(lambda n: float(n), goal_info))
            if goal_info[0] > self.max_goal_distance:
                goal_info = [self.max_goal_distance, goal_info[1]]

        # ボールが見えてるか
        if ball_info is None:
            ball_info = [self.max_ball_distance + 1, self.max_ball_angle + 1]
        else:
            ball_info = list(map(lambda n: float(n), ball_info))
            if ball_info[0] > self.max_ball_distance:
                ball_info = [self.max_ball_distance, ball_info[1]]
            elif ball_info[0] < 5:
                self.ball_damage -= (1000 + self.ball_damage) / 20
            self.ball_distance = ball_info[0]

        if self.goal_distance > goal_info[0]:
            reward_see += 200
            if self.ball_distance > ball_info[0]:
                reward_see *= 2.5
                self.ball_damage -= (1000 + self.ball_damage) / 30
        self.goal_distance = goal_info[0]
        self.ball_distance = ball_info[0]

        # ログの追加
        reward += reward_see + reward_action - self.ball_damage
        self._update(reward, reward_see, reward_action, self.ball_damage * -1, self.goal)
        """=========ここまで==========="""
        if self.time % self.goal_reset == 0:
            self.goal = 0
        return (int(ball_info[0] / self.ball_distance_interval), int(ball_info[1] / self.ball_angle_interval),
                int(goal_info[0] / self.goal_distance_interval), int(goal_info[1] / self.goal_angle_interval)
                ), reward, end_action
