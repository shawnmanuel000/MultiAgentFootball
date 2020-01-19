models=([0]="ddqn" [1]="ddpg" [2]="ppo" [3]="rand")

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		xterm -display ":0" -e $script $2 # Add -hold argument after xterm to debug
	fi
}

for i in "${!models[@]}"
do
	open_terminal "bash train.sh 100000 ${models[$i]} $(( 1000*$i ))"
done