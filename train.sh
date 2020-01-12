models=([0]="ddqn" [1]="ddpg" [2]="ppo")

model=$1
baseport=$2
workers=16
steps=10000

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		xterm -hold -display ":0" -e $script $2 # Add -hold argument after xterm to debug
	fi
}

run()
{
	numWorkers=$1
	steps=$2
	agent=$3

	echo "Workers: $numWorkers, Steps: $steps, Agent: $agent"
	
	ports=()
	for j in `seq 1 $numWorkers` 
	do
		port=$((8000+$baseport+$j))
		ports+=($port)
		open_terminal "python3 -B train.py --selfport $port" &
	done

	sleep 4
	port_string=$( IFS=$' '; echo "${ports[*]}" )
	open_terminal "python3 -B train.py --steps $steps --model $agent --workerports $port_string"
}

run $workers $steps $model

