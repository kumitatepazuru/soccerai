import libscips.signal
import itertools
from matplotlib import pyplot as plt, animation


class env(libscips.signal.player_signal):
    def __init__(self, ADDRESS="127.0.0.1", HOST="", send_log=False, recieve_log=False,
                 analysis_log=("unknown", "init", "error", "hear"), max_boal_distance=40, min_boal_distance=0,
                 max_boal_angle=100,
                 min_boal_angle=-100, max_goal_distance=55, min_goal_distance=5, max_goal_angle=100,
                 min_goal_angle=-100):
        # 初期設定
        super().__init__(ADDRESS, HOST, send_log, recieve_log, analysis_log)
        self.actions = ["dash", "kick", "turn1", "turn2", "turn3", "turn4", "turn5", "kick2", "kick3"]
        self.max_ball_distance = max_boal_distance
        self.min_ball_distance = min_boal_distance
        self.max_ball_angle = max_boal_angle
        self.min_ball_angle = min_boal_angle
        self.max_goal_distance = max_goal_distance
        self.min_goal_distance = min_goal_distance
        self.max_goal_angle = max_goal_angle
        self.min_goal_angle = min_goal_angle
        self.turn_repeat = [0, 0]
        self.team = None
        self.enemy = None
        self.kick_time = 0
        self.reward_log = [0]

    def __len__(self):
        # len(env)をやったときに返すデータ
        return len(self.actions)

    def _update(self, _, _2, _3):
        """グラフを更新するための関数"""
        # 現在のグラフを消去する
        plt.cla()
        # 折れ線グラフを再描画する
        plt.plot(list(range(len(self.reward_log))), self.reward_log)

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
        return ball_info[0], ball_info[1], self.turn_repeat[0]

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
        self.kick_time += 1
        reward = 0
        if action == 0:
            reward -= self.repeat(0, out=20)
            self.send_dash(100)
        elif action == 1:
            self.send_kick(100, 0)
        elif action == 2:
            reward -= self.repeat(1)
            self.send_turn(-30)
        elif action == 3:
            reward -= self.repeat(2)
            self.send_turn(30)
        elif action == 4:
            reward -= self.repeat(3)
            self.send_turn(-10)
        elif action == 5:
            reward -= self.repeat(4)
            self.send_turn(10)
        elif action == 6:
            reward -= self.repeat(5, out=2)
            self.send_turn(120)
        elif action == 7:
            self.send_kick(50, 0)
        elif action == 8:
            self.send_kick(25, 0)

        # 見えているものを確認する
        rec = {"type": None}
        rec_raw = []
        while rec["type"] != "see":
            rec_raw = self.recieve_msg()
            rec = self.msg_analysis(rec_raw)
            if rec["type"] == "hear":
                # ゴールしたときの報酬（できたらめっちゃ褒める、できなかったらめっちゃ怒ってもらう）
                if rec["contents"][:6] == "goal_" + self.team:
                    reward += 1000
                elif rec["contents"][:5] == "goal_" + self.enemy:
                    reward -= 1000
                # kick_in等のPKになったら怒ってもらう
                if rec["contents"][:9] == "kick_in_" + self.enemy or \
                        rec["contents"][:11] == "goal_kick_" + self.enemy or \
                        rec["contents"][:13] == "corner_kick_" + self.enemy or \
                        rec["contents"][:11] == "free_kick_" + self.enemy:
                    reward -= 200
        ball_info = self.see_analysis(rec_raw, "b")

        b = 0
        # ボールが見えてるか
        if ball_info is None:
            ball_info = [self.max_ball_distance, self.max_ball_angle]
        else:
            b = 1
            ball_info = list(map(lambda n: float(n), ball_info))
            # 距離に応じた報酬をもらう
            reward += 80 - (ball_info[0] * 2)
        # キックをしたとき
        if action == 1 or action == 7 or action == 8:
            # ボールがあるなら褒めてもらう。なかったら怒ってもらう。（kickしまくり防止）
            if ball_info[0] > 1:
                reward -= 100
            else:
                reward += 100
        # エラー回避
        if ball_info[0] >= 40:
            ball_info[0] = 39.9
        reward -= 0.1

        goal_info = self.see_analysis(rec_raw, ["g", self.enemy])
        if goal_info is None:
            # ゴールとボールが見えてるならボーナスをあげる
            if b == 1:
                reward += 10
        # print( "\033[38;5;12m[INFO]\t\033[38;5;13mreward \033[4m" + str(reward) +
        #        "\033[0m\033[38;5;10m\taction\033[4m" + self.actions[action] + "\033[0m")

        # ログの追加
        self.reward_log.append(reward)
        return (int(ball_info[0]), int(ball_info[1])), reward, False

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
            'interval': 100,  # 更新間隔 (ミリ秒)
            'frames': itertools.count(0, 1),  # フレーム番号を無限に生成するイテレータ
        }
        _ = animation.FuncAnimation(**params)

        # グラフを表示する
        plt.show()
