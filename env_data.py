import random

import libscips.player
import numpy as np


def logtoplot(data, interval):
    return list(map(lambda n: n.mean(), np.array_split(np.array(data), interval)))


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
        # self.logger1 = TensorBoardLogger(log_dir='./' + logdir + '/reward')
        # self.logger2 = TensorBoardLogger(log_dir='./' + logdir + '/see')
        # self.logger3 = TensorBoardLogger(log_dir='./' + logdir + '/action')
        # self.logger4 = TensorBoardLogger(log_dir='./'+logdir+'/damage')
        # self.logger5 = TensorBoardLogger(log_dir='./' + logdir + '/goal')

        self.max_ball_distance = max_ball_distance
        self.max_ball_angle = max_ball_angle
        self.max_goal_distance = max_goal_distance
        self.max_goal_angle = max_goal_angle
        self.actions = actions
        self.kickcount = 0
        self.damage = 0
        self.team = None
        self.enemy = None
        self.turn_repeat = [0, 0]

    def __len__(self):
        # len(env)をやったときに返すデータ
        return self.actions

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
        # elif float(ball_info[0]) < self.special_division_value and self.special_division:
        #     ball_info = [self.max_ball_distance + 2, float(ball_info[1])]
        if goal_info is None:
            goal_info = [self.max_goal_distance + 1, self.max_goal_angle + 1]
        # if float(goal_info[0]) > self.max_goal_distance:
        #     goal_info = [self.max_goal_distance, goal_info[1]]
        return float(ball_info[0]), float(ball_info[1]), float(goal_info[0]), float(goal_info[1])

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

    def random(self):
        return random.randint(0, self.actions - 1)

    def step(self, action):
        self.damage -= 0.001
        # actionから移動をして報酬を計算するプログラム
        reward = 0
        # self.goal_damage += 0.01
        # self.ball_damage += (12 - self.num) * 0.1 * (self.time - self.kick_time) * ((self.time - self.goal_time) / 10)
        if action == 0:
            self.send_dash(100)
            # reward -= self.repeat(1, out=20)
        elif action == 1:
            self.send_turn(10)
            # reward -= self.repeat(1)
        elif action == 2:
            self.send_turn(-10)
            # reward -= self.repeat(1)
        elif action == 3:
            self.send_turn(30)
            # reward -= self.repeat(1)
        elif action == 4:
            self.send_turn(-30)
            # reward -= self.repeat(1)
        elif action == 5:
            self.send_kick(100, 0)
            reward -= self.repeat(1)

        """==========reward setting=========="""
        # 見えているものを確認する
        rec = {"type": None}
        rec_raw = []
        done = False

        while rec["type"] != "see":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)
            if rec["type"] == "hear":
                if rec["contents"] == "half_time":
                    done = True
                    self.send_move(-10,-10)
                # ゴールしたときの報酬（できたらめっちゃ褒める）
                # if rec["contents"][:6] == "goal_" + self.team:
                #     self.send_move(-10, -10)
                #     self.ball_damage = -1000 * int(rec["contents"][7:])
                #     reward_action += 1000 * int(rec["contents"][7:])
                #     self.goal += 1
                #     self.goal_time = self.time
                #     print("GOAL!!!!!!", int(rec["contents"][7:]), "time:", rec["time"])
            if rec["type"] == "sense_body":
                if self.kickcount < int(rec["value"][4][1]):
                    #         if self.ball_damage / 100 > 500:
                    #             self.ball_damage -= self.ball_damage / 5
                    #         else:
                    #             self.ball_damage -= 500
                    #         reward_action += 100
                    reward += 1
                    self.damage = 0
                    print("kick!!!!", "time:", rec["time"])
                    #         self.kick_time = self.time
                    self.kickcount = int(rec["value"][4][1])

        ball_info = self.see_analysis(rec_raw, "b")
        goal_info = self.see_analysis(rec_raw, ["g", self.enemy])

        # ゴールが見えているか
        if goal_info is None:
            goal_info = [self.max_goal_distance + 1, self.max_goal_angle + 1]
        # else:
        #     goal_info = list(map(lambda n: float(n), goal_info))
        #     if goal_info[0] > self.max_goal_distance:
        #         goal_info = [self.max_goal_distance, goal_info[1]]
        #
        # # ボールが見えてるか
        if ball_info is None:
            ball_info = [self.max_ball_distance + 1, self.max_ball_angle + 1]
        elif float(ball_info[0]) < 5:
            self.damage -= self.damage / 5
        #
        # if self.goal_distance > goal_info[0]:
        #     if self.ball_distance > ball_info[0]:
        #         self.ball_damage -= self.ball_damage / 30
        #         reward_see *= 2.5
        #
        # self.goal_distance = goal_info[0]
        # self.ball_distance = ball_info[0]

        # ログの追加
        # reward += reward_see + reward_action - self.ball_damage
        # reward += reward_see + reward_action
        # self._update(reward, reward_see, reward_action, self.ball_damage * -1, self.goal)
        # """=========ここまで==========="""
        # if self.time % self.goal_reset == 0:
        #     self.goal = 0
        # return (int(ball_info[0] / self.ball_distance_interval), int(ball_info[1] / self.ball_angle_interval),
        #         int(goal_info[0] / self.goal_distance_interval), int(goal_info[1] / self.goal_angle_interval)
        #         ), reward, end_action
        return (float(ball_info[0]), float(ball_info[1]), float(goal_info[0]),
                float(goal_info[1])), done, reward - self.damage
