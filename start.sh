trap killall SIGKILL
killall(){
  kill 0
}
rcssserver server::simulator_step=50 server::game_logging=false server::text_logging=false server::nr_normal_halfs=1 server::nr_extra_halfs=0 server::penalty_shoot_outs=false server::half_time=-1 server::stamina_capacity=-1 server::stamina_max=99999 server::auto_mode=true server::drop_ball_time=500 &
sleep 1
soccerwindow2-qt4 &
python3 main.py $1 $2
wait