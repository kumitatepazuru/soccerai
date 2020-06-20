rm ./logdir -rf
trap killall SIGKILL
killall(){
  kill 0
}
rcssserver server::game_logging=false server::text_logging=false server::nr_normal_halfs=1 server::use_offside=off server::synch_mode=true server::free_kick_faults=false server::forbid_kick_off_offside=false server::nr_extra_halfs=0 server::penalty_shoot_outs=false server::half_time=-1 server::stamina_capacity=-1 server::auto_mode=true server::drop_ball_time=500 &
soccerwindow2-qt4 &
#cd agent2d-3.1.1/src/
#./start.sh -n 1 -u 1
#cd ../../
python3 main.py $1 $2 &
firefox localhost:6006 &
tensorboard --logdir=./logdir
wait