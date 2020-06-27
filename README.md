# soccerai
## How to install
```shell script
pip3 install -r requirement.txt
```
## How to run the program
```shell script
python3 main.py [team_name_1] [team_name_2]
```

## Argument description
```
usage: main.py [-h] [--server_port SERVER_PORT] [-f1 FILE1] [-f2 FILE2] [-o]
               [--player_num_1 PLAYER_NUM_1] [--player_num_2 PLAYER_NUM_2]
               [-l] [-s] [-w] [--gamma GAMMA] [--learning_rate LEARNING_RATE]
               [--save_interval SAVE_INTERVAL] [--address ADDRESS]
               [--host HOST] [--send_log] [--recieve_log]
               [--analysis_log [ANALYSIS_LOG [ANALYSIS_LOG ...]]]
               [--logdir LOGDIR] [--max_ball_distance MAX_BALL_DISTANCE]
               [--min_ball_distance MIN_BALL_DISTANCE]
               [--ball_distance_interval BALL_DISTANCE_INTERVAL]
               [--special_division]
               [--special_division_value SPECIAL_DIVISION_VALUE]
               [--damage_reset_ball_distance DAMAGE_RESET_BALL_DISTANCE]
               [--max_ball_angle MAX_BALL_ANGLE]
               [--min_ball_angle MIN_BALL_ANGLE]
               [--ball_angle_interval BALL_ANGLE_INTERVAL]
               [--max_goal_distance MAX_GOAL_DISTANCE]
               [--min_goal_distance MIN_GOAL_DISTANCE]
               [--goal_distance_interval GOAL_DISTANCE_INTERVAL]
               [--max_goal_angle MAX_GOAL_ANGLE]
               [--min_goal_angle MIN_GOAL_ANGLE]
               [--goal_angle_interval GOAL_ANGLE_INTERVAL] [--noise NOISE]
               [--actions ACTIONS] [--goal_reset GOAL_RESET]
               team_name_1 team_name_2

A program for learning agent in a robocup 2D simulation league

positional arguments:
  team_name_1           Specify a team name for Team 1
  team_name_2           Specify a team name for Team 2

optional arguments:
  -h, --help            show this help message and exit
  --server_port SERVER_PORT
                        Specifies the server port to connect to. If you
                        specify --auto_start argument, the server port will be
                        set to this value, and the It is.
  -f1 FILE1, --file1 FILE1
                        Specify the trained data for team 1
  -f2 FILE2, --file2 FILE2
                        Specify the trained data for team 2
  -o, --only_team_1     Register one team only
  --player_num_1 PLAYER_NUM_1
                        Set the number of players registered for Team 1
  --player_num_2 PLAYER_NUM_2
                        Set the number of players registered for Team 2
  -l, --logging         Save the trained data
  -s, --auto_server     Auto-start the server.
  -w, --auto_window     Auto-start soccerwindow2 (must be installed
                        beforehand).
  --gamma GAMMA         Specifies the gamma of the ActorCritic.train function
  --learning_rate LEARNING_RATE
                        Specify the learning rate of soccerai
  --save_interval SAVE_INTERVAL
                        Specify the storage interval of learned data in
                        episode.
  --address ADDRESS     The IP address of the server to connect to
  --host HOST           Enter the host's address
  --send_log            Display the logs you sent to the server
  --recieve_log         Display the logs received from the server
  --analysis_log [ANALYSIS_LOG [ANALYSIS_LOG ...]]
                        Display the log of messages received from the server
                        (specify multiple messages in the filter)
  --logdir LOGDIR       Specify the location of the tensorboard's log file
                        (the data will be deleted if the specified location
                        exists)
  --max_ball_distance MAX_BALL_DISTANCE
                        Specifies the maximum distance of the ball that the AI
                        can recognize. If the maximum value is exceeded, it
                        will be cut down to the maximum value.
  --min_ball_distance MIN_BALL_DISTANCE
                        Specifies the minimum distance of the ball that the AI
                        can recognize.
  --ball_distance_interval BALL_DISTANCE_INTERVAL
                        The distance specifies how ambiguous the number
                        between the distances of the ball is to be learned.
  --special_division    Create a dedicated Q table index when a ball is seen
                        in a specific range
  --special_division_value SPECIAL_DIVISION_VALUE
                        Specify a specific range of special_division
  --damage_reset_ball_distance DAMAGE_RESET_BALL_DISTANCE
                        Specify the distance of the ball as a condition for
                        resetting the damage you receive per episode.
  --max_ball_angle MAX_BALL_ANGLE
                        Specifies the maximum angle relative to the ball that
                        the AI can recognize. If the relative angle is greater
                        than the maximum value, it will be cut down to the
                        maximum value.
  --min_ball_angle MIN_BALL_ANGLE
                        Specifies the minimum relative angle to the ball that
                        the AI can recognize. If the relative angle is smaller
                        than the minimum value, it is rounded up to the
                        minimum value.
  --ball_angle_interval BALL_ANGLE_INTERVAL
                        The angle specifies how ambiguous the relative angle
                        of the ball is to be learned.
  --max_goal_distance MAX_GOAL_DISTANCE
                        Specifies the maximum angle relative to the goal that
                        the AI can recognize. If the relative angle is greater
                        than the maximum value, it will be cut down to the
                        maximum value.
  --min_goal_distance MIN_GOAL_DISTANCE
                        Specifies the minimum relative angle to the goal that
                        the AI can recognize. If the relative angle is smaller
                        than the minimum value, it is rounded up to the
                        minimum value.
  --goal_distance_interval GOAL_DISTANCE_INTERVAL
                        The angle specifies how ambiguous the relative angle
                        of the goal is to be learned.
  --max_goal_angle MAX_GOAL_ANGLE
                        Specifies the maximum angle relative to the goal that
                        the AI can recognize. If the relative angle is greater
                        than the maximum value, it will be cut down to the
                        maximum value.
  --min_goal_angle MIN_GOAL_ANGLE
                        Specifies the minimum relative angle to the goal that
                        the AI can recognize. If the relative angle is smaller
                        than the minimum value, it is rounded up to the
                        minimum value.
  --goal_angle_interval GOAL_ANGLE_INTERVAL
                        The angle specifies how ambiguous the relative angle
                        of the goal is to be learned.
  --noise NOISE         How much of the actions given by the AI will be used
  --actions ACTIONS     Specify the number of actions
  --goal_reset GOAL_RESET
                        Specify the specific episode in the graph that shows
                        the number of times a goal is scored within a specific
                        episode displayed on tensorboard.

```