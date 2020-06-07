import random

import libscips.signal
import itertools
import numpy as np
from matplotlib import pyplot as plt, animation


def logtoplot(data, interval):
    return list(map(lambda n: n.mean(), np.array_split(np.array(data), interval)))


class env(libscips.signal.player_signal):
    def __init__(self, ADDRESS="127.0.0.1", HOST="", send_log=False, recieve_log=False,
                 analysis_log=("unknown", "init", "error","hear"),
                 max_ball_distance=40, min_ball_distance=0, ball_distance_interval=10,
                 max_ball_angle=100, min_ball_angle=-100, ball_angle_interval=40,
                 max_goal_distance=55, min_goal_distance=5, goal_distance_interval=10,
                 max_goal_angle=100, min_goal_angle=-100, goal_angle_interval=40,
                 report_interval=50, noise=0.8):
        # 初期設定
        super().__init__(ADDRESS, HOST, send_log, recieve_log, analysis_log)
        self.actions = ["dash", "kick", "turn1", "turn2", "turn3", "turn4", "turn5", "kick2", "kick3"]
        self.report_interval = report_interval
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
        self.noise = noise
        self.turn_repeat = [0, 0]
        self.team = None
        self.enemy = None
        self.reward_log = [0]
        self.reward_repeat_log = [0]
        self.reward_action_log = [0]
        self.reward_see_log = [0]

    def __len__(self):
        # len(env)をやったときに返すデータ
        return len(self.actions)

    def _update(self, _, _2, _3):
        """グラフを更新するための関数"""

        split_num = int(len(self.reward_log) / self.report_interval) + 1

        # 現在のグラフを消去する
        plt.cla()
        # 折れ線グラフを再描画する
        plt.plot(list(range(split_num)), logtoplot(self.reward_log, split_num), "-o", label="reward")
        plt.plot(list(range(split_num)), logtoplot(self.reward_repeat_log, split_num), "-o",
                 label="repeat_reward")
        plt.plot(list(range(split_num)), logtoplot(self.reward_action_log, split_num), "-o",
                 label="action_reward")
        plt.plot(list(range(split_num)), logtoplot(self.reward_see_log, split_num), "-o",
                 label="see_reward")
        plt.text(0.8, 0.8, "episode:"+str(len(self.reward_log)), size = 15)
        plt.legend()

    def reset(self, name):
        # 環境のリセット
        self.team = self.send_init(name)[0][1]
        self.enemy = "l" * (self.team == "r") + "r" * (self.team == "l")
        self.send_move(-10, -10)
        rec = {"type": None}
        rec_raw = []
        while rec["type"] != "see":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)
        ball_info = self.see_analysis(rec_raw, "b")
        if ball_info is None:
            ball_info = (self.max_ball_distance, self.max_ball_angle)
        return int(float(ball_info[0]) / self.ball_distance_interval), int(float(ball_info[1]) /
                                                                    self.ball_angle_interval), self.turn_repeat[0]

    def repeat(self, num, out=10):
        # 頭がもげるのを防止するプログラム（まだ自爆しようとするagentがいる）
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
        # actionから移動をして報酬を計算するプログラム
        reward = 0
        reward_repeat = 0
        reward_action = 0
        reward_see = 0
        if random.random() < self.noise:
            end_action = action
            if action == 0:
                reward_repeat -= self.repeat(0, out=20)
                self.send_dash(100)
            elif action == 1:
                self.send_kick(100, 0)
            elif action == 2:
                reward_repeat -= self.repeat(1)
                self.send_turn(-30)
            elif action == 3:
                reward_repeat -= self.repeat(1)
                self.send_turn(30)
            elif action == 4:
                reward_repeat -= self.repeat(1)
                self.send_turn(-10)
            elif action == 5:
                reward_repeat -= self.repeat(1)
                self.send_turn(10)
            elif action == 6:
                reward_repeat -= self.repeat(1, out=2)
                self.send_turn(120)
            elif action == 7:
                self.send_kick(50, 0)
            elif action == 8:
                self.send_kick(25, 0)
        else:
            action = random.randrange(0,len(self.actions))
            end_action = action

        # 見えているものを確認する
        rec = {"type": None}
        rec_raw = []
        while rec["type"] != "see":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)
        #     if rec["type"] == "hear":
        #         # ゴールしたときの報酬（できたらめっちゃ褒める、できなかったらめっちゃ怒ってもらう）
        #         if rec["contents"][:6] == "goal_" + self.team:
        #             self.send_move(-10, -10)
        #             reward += 1000
        #         elif rec["contents"][:6] == "goal_" + self.enemy:
        #             self.send_move(-10, -10)
        #             reward -= 1000
        #         # kick_in等のPKになったら怒ってもらう
        #         if rec["contents"][:9] == "kick_in_" + self.enemy or \
        #                 rec["contents"][:11] == "goal_kick_" + self.enemy or \
        #                 rec["contents"][:13] == "corner_kick_" + self.enemy or \
        #                 rec["contents"][:11] == "free_kick_" + self.enemy:
        #             reward -= 200
        ball_info = self.see_analysis(rec_raw, "b")

        # ボールが見えてるか
        if ball_info is None:
            ball_info = [self.max_ball_distance, self.max_ball_angle]
        else:
            ball_info = list(map(lambda n: float(n), ball_info))
            if ball_info[0] > self.max_ball_distance:
                ball_info = [self.max_ball_distance,ball_info[1]]
            elif ball_info[0] < 1:
                reward_see += 25
            # 距離に応じた報酬をもらう
            reward_see += 80 - (ball_info[0] * 2)
        if action == 1 or action == 7 or action == 8:
            if ball_info[0] < 1:
                print("INFO: agent kicked the ball!")
                reward_action += 100
            else:
                reward_action -= 100

        # ログの追加
        reward += reward_repeat + reward_action + reward_see
        self.reward_log.append(reward)
        self.reward_action_log.append(reward_action)
        self.reward_repeat_log.append(reward_repeat)
        self.reward_see_log.append(reward_see)
        return (int(ball_info[0] / self.ball_distance_interval),
                int(ball_info[1] / self.ball_angle_interval)), reward, end_action

    def show_log(self):
        # ログの表示
        # 描画領域
        fig = plt.figure(figsize=(10, 6))
        # 描画するデータ
        x = []
        y = []

        params = {
            'fig': fig,
            'func': self._update,  # グラフを更新する関数
            'fargs': (x, y),  # 関数の引数 (フレーム番号を除く)
            'interval': self.report_interval*10,  # 更新間隔 (ミリ秒)
            'frames': itertools.count(0, 1),  # フレーム番号を無限に生成するイテレータ
        }
        _ = animation.FuncAnimation(**params)

        # グラフを表示する
        plt.show()
