from socket import socket, AF_INET, SOCK_DGRAM
from sexpdata import Symbol, loads


def func(text):
    if type(text) == Symbol:
        return text.value()
    elif type(text) == list:
        return tostring(text)
    else:
        return str(text)


def tostring(text):
    return list(map(func, text))


class player_signal:
    def __init__(self, ADDRESS="127.0.0.1", HOST="", SERVER_PORT=6000, send_log=False, recieve_log=False,
                 analysis_log=("unknown","init", "error")):
        self.ADDRESS = ADDRESS
        self.s = socket(AF_INET, SOCK_DGRAM)
        ok = 0
        i = 0
        print("\033[38;5;12m[INFO]\t\033[38;5;13mSearching for available ports ...\033[0m")
        while ok == 0:
            try:
                self.s.bind((HOST, 1000 + i))
                ok = 1
            except OSError:
                i += 1
        self.recieve_port = 1000 + i
        self.recieve_log = recieve_log
        self.send_log = send_log
        self.analysis_log = analysis_log
        self.no = ""
        self.SERVER_PORT = SERVER_PORT
        self.player_port = 0
        self.error = {"no more player or goalie or illegal client version": 0, "unknown command": 1}

    def __del__(self):
        self.s.close()

    def send_msg(self, text, PORT=None, log=None):
        if PORT is None:
            PORT = self.SERVER_PORT
        self.s.sendto((text + "\0").encode(), (self.ADDRESS, PORT))
        self.send_logging(text, PORT, log=log)

    def send_logging(self, text, PORT, log=None):
        if log is None:
            log = self.send_log
        if log:
            print("\033[38;5;12m[INFO]\t" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (self.no != "") + "\033[38;5;10mSend msg.\t" +
                  "\033[38;5;9mPORT \033[4m" + str(self.recieve_port) + "\033[0m\033[38;5;9m → \033[4m" + str(PORT) +
                  "\033[0m\t\033[38;5;6mTEXT \033[4m" + text + "\033[0m")

    def send_init(self, name, goalie=False, version=15, log=None):
        msg = "(init " + name + " (goalie)" * goalie + " (version " + str(version) + "))"
        self.send_msg(msg, log=log)
        r = self.recieve_msg(log=log)
        self.player_port = r[1][1]
        return r

    def send_move(self, x, y, log=None):
        msg = "(move " + str(x) + " " + str(y) + ")"
        self.send_msg(msg, self.player_port, log=log)

    def send_dash(self, power, log=None):
        msg = "(dash " + str(power) + ")"
        self.send_msg(msg, self.player_port, log=log)

    def send_turn(self, moment, log=None):
        msg = "(turn " + str(moment) + ")"
        self.send_msg(msg, self.player_port, log=log)

    def send_turn_neck(self, angle, log=None):
        msg = "(turn_neck " + str(angle) + ")"
        self.send_msg(msg, self.player_port, log=log)

    def send_kick(self, power, direction, log=None):
        msg = "(kick " + str(power) + " " + str(direction) + ")"
        self.send_msg(msg, self.player_port, log=log)

    def recieve_msg(self, log=None):
        msg, address = self.s.recvfrom(8192)
        msg = tostring(loads(msg[:-1].decode("utf-8")))
        if msg[0] == "think":
            if log is None:
                log = self.recieve_log
            if log:
                print("\033[38;5;12m[INFO]" + (
                        "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                              self.no != "") + "\t\033[0m\033[38;5;10mGet \033[4mthink\033[0m msg.\t\033[38;5;9mPORT "
                                               "\033[4m" + str(self.recieve_port) + "\033[0m\033[38;5;9m ← \033[4m" +
                      str(address[1]) + "\033[0m\t\033[38;5;6mIP \033[4m" + address[0] + "\033[0m")
            self.send_msg("(done)",self.player_port)
            return self.recieve_msg(log)
        else:
            if log is None:
                log = self.recieve_log
            if log:
                print("\033[38;5;12m[INFO]" + (
                        "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                              self.no != "") + "\t\033[0m\033[38;5;10mGet msg.\t\033[38;5;9mPORT \033[4m" + str(
                    self.recieve_port) + "\033[0m\033[38;5;9m ← \033[4m" +
                      str(address[1]) + "\033[0m\t\033[38;5;6mIP \033[4m" + address[0] + "\033[0m")
            return msg, address

    def msg_analysis(self, text, log_show=None):
        text = text[0]
        if text[0] == "error":
            text = text[1].replace("_", " ")
            log = "\033[38;5;1m[ERR]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[4m" + text + "\033[0m"
            r = {"type": "error", "value": str(self.error.get(text)) + str(self.error.get(text) is None)}
        elif text[0] == "init":
            self.no = text[2]
            log = "\033[38;5;10m[OK]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[38;5;10minit msg.\t\033[4m" + "\033[38;5;11mleft team" * (
                          text[1] == "l") + \
                  "\033[38;5;1mright team" * (text[1] == "r") + "\033[0m\033[38;5;6m no \033[4m" + text[2] + "\033[0m"
            r = {"type": "init", "value": text[:-2]}
        elif text[0] == "server_param" or text[0] == "player_param" or text[0] == "player_type":
            log = "\033[38;5;12m[INFO]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[38;5;10m" + text[0] + " msg.\033[0m"
            r = {"type": text[0], "value": text[1:]}
        elif text[0] == "see" or text[0] == "sense_body":
            log = "\033[38;5;12m[INFO]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[38;5;10m" + text[0] + " msg. \033[38;5;9mtime \033[4m" + text[
                      1] + "\033[0m"
            r = {"type": text[0], "time": int(text[1]), "value": text[2:]}
        elif text[0] == "hear":
            log = "\033[38;5;12m[INFO]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[38;5;10mhear msg. \033[38;5;9mtime \033[4m" + text[1] + "\033[0m " +\
                  "\033[38;5;6mspeaker \033[4m" + text[2] + "\033[0m " + "\033[38;5;13mcontents \033[4m" + text[3] + \
                  "\033[0m"
            r = {"type": "hear", "time": int(text[1]), "speaker": text[2], "contents": text[3]}
        elif text[0] == "change_player_type":
            log = "\033[38;5;12m[INFO]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[38;5;10mhear msg. \033[0m"
            r = {"type": "change_player_type", "value": text[1]}
        else:
            log = "\033[38;5;12m[INFO]" + (
                    "\033[38;5;13mno \033[4m" + self.no + "\033[0m ") * (
                          self.no != "") + "\t\033[38;5;10mUnknown return value \033[0m\033[4m" + str(text) + "\033[0m"
            r = {"type": "unknown", "value": text}
        if log_show is None:
            log_show = r["type"] in self.analysis_log
        if log_show:
            print(log)
        return r

    def see_analysis(self, text, hit, log_show=None):
        if type(hit) == str:
            hit = [hit]
        text = text[0]
        for i in text[2:]:
            if i[0] == hit:
                if log_show is None:
                    log_show = hit in self.analysis_log or hit[0] in self.analysis_log
                if log_show:
                    print("\033[38;5;12m[INFO]\t\033[38;5;10mThere was a " + str(
                        hit) + " in the visual information.\033[0m")
                return i[1:]
        if log_show:
            print("\033[38;5;12m[INFO]\t\033[38;5;10mThere was no " + str(hit) + " in the visual information.\033[0m")
        return None
