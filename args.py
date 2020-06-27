import argparse


def args():
    parser = argparse.ArgumentParser(description='A program for learning agent in a robocup 2D simulation league')
    parser.add_argument('--server_port',
                        help='Specifies the server port to connect to. If you specify --auto_start argument, '
                             'the server port will be set to this value, and the It is.',
                        default=6000, type=int)
    parser.add_argument("-f1", "--file1", help="Specify the trained data for team 1")
    parser.add_argument("-f2", "--file2", help="Specify the trained data for team 2")
    parser.add_argument("-o", "--only_team_1", help="Register one team only", action='store_true')
    parser.add_argument("-i", "--auto_ip", help="Automatically assign IP addresses when using auto_server", action='store_true')
    parser.add_argument("--player_num_1", help="Set the number of players registered for Team 1",
                        default=1, type=int)
    parser.add_argument("--player_num_2", help="Set the number of players registered for Team 2",
                        default=1, type=int)
    parser.add_argument("-l", "--logging", help="Save the trained data", action='store_true')
    parser.add_argument("-s", "--auto_server", help="Auto-start the server.", action='store_true')
    parser.add_argument("-w", "--auto_window", help="Auto-start soccerwindow2 (must be installed beforehand).",
                        action='store_true')
    parser.add_argument("team_name_1", help="Specify a team name for Team 1")
    parser.add_argument("team_name_2", help="Specify a team name for Team 2")
    parser.add_argument("--gamma", help="Specifies the gamma of the ActorCritic.train function", default=0.95,
                        type=float)
    parser.add_argument("--learning_rate", help="Specify the learning rate of soccerai", default=0.1, type=float)
    parser.add_argument("--save_interval", help="Specify the storage interval of learned data in episode.",
                        default=25000, type=int)

    # Here are the arguments to the env_data.env class
    parser.add_argument("--address", help="The IP address of the server to connect to",
                        default="127.0.0.1")
    parser.add_argument("--host", help="Enter the host's address",
                        default="")
    parser.add_argument("--send_log", help="Display the logs you sent to the server", action='store_true')
    parser.add_argument("--recieve_log", help="Display the logs received from the server", action='store_true')
    parser.add_argument("--analysis_log", help="Display the log of messages received from the server "
                                               "(specify multiple messages in the filter)", nargs="*",
                        default=("unknown", "init", "error"))
    parser.add_argument("--logdir", help="Specify the location of the tensorboard's log file (the data will be deleted "
                                         "if the specified location exists)", default="logdir")
    parser.add_argument("--max_ball_distance", help="Specifies the maximum distance of the ball that the AI can "
                                                    "recognize. If the maximum value is exceeded, it will be cut down "
                                                    "to the maximum value.", default=40, type=int)
    parser.add_argument("--min_ball_distance", help="Specifies the minimum distance of the ball that the AI can "
                                                    "recognize.", default=0, type=int)
    parser.add_argument("--ball_distance_interval", help="The distance specifies how ambiguous the number between the "
                                                         "distances of the ball is to be learned.", default=10, type=int
                        )
    parser.add_argument("--special_division", help="Create a dedicated Q table index when a ball is seen in a specific "
                                                   "range", action='store_true', default=False)
    parser.add_argument("--special_division_value", help="Specify a specific range of special_division", default=1,
                        type=int)
    parser.add_argument("--damage_reset_ball_distance", help="Specify the distance of the ball as a condition for"
                                                             " resetting the damage you receive per episode.",
                        default=5, type=int)
    parser.add_argument("--max_ball_angle", help="Specifies the maximum angle relative to the ball that the AI can "
                                                 "recognize. If the relative angle is greater than the maximum value, "
                                                 "it will be cut down to the maximum value.", default=100, type=int)
    parser.add_argument("--min_ball_angle", help="Specifies the minimum relative angle to the ball that the AI can "
                                                 "recognize. If the relative angle is smaller than the minimum value, "
                                                 "it is rounded up to the minimum value.", default=-100, type=int)
    parser.add_argument("--ball_angle_interval", help="The angle specifies how ambiguous the relative angle "
                                                      "of the ball is to be learned.", default=40, type=int)
    parser.add_argument("--max_goal_distance", help="Specifies the maximum angle relative to the goal that the AI can "
                                                    "recognize. If the relative angle is greater than the maximum "
                                                    "value, it will be cut down to the maximum value.", default=55,
                        type=int)
    parser.add_argument("--min_goal_distance", help="Specifies the minimum relative angle to the goal that the AI can "
                                                    "recognize. If the relative angle is smaller than the minimum "
                                                    "value, it is rounded up to the minimum value.", default=5, type=int
                        )
    parser.add_argument("--goal_distance_interval", help="The angle specifies how ambiguous the relative angle "
                                                         "of the goal is to be learned.", default=10, type=int)
    parser.add_argument("--max_goal_angle", help="Specifies the maximum angle relative to the goal that the AI can "
                                                 "recognize. If the relative angle is greater than the maximum value, "
                                                 "it will be cut down to the maximum value.", default=100, type=int)
    parser.add_argument("--min_goal_angle", help="Specifies the minimum relative angle to the goal that the AI can "
                                                 "recognize. If the relative angle is smaller than the minimum value, "
                                                 "it is rounded up to the minimum value.", default=-100, type=int)
    parser.add_argument("--goal_angle_interval", help="The angle specifies how ambiguous the relative angle "
                                                      "of the goal is to be learned.", default=50, type=int)
    parser.add_argument("--noise", help="How much of the actions given by the AI will be used", default=0.5, type=float)
    parser.add_argument("--actions", help="Specify the number of actions", default=6, type=int)
    parser.add_argument("--goal_reset", help="Specify the specific episode in the graph that shows the number of times "
                                             "a goal is scored within a specific episode displayed on tensorboard.",
                        default=10000, type=int)
    return parser.parse_args()
