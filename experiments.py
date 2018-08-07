import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from src.helpers import run_with_mahi_settings, get_open_udp_port
from src.senders import Sender
from src.ml_strategy import ReinforcementStrategy
from src.ml_helpers import LSTM_DQN

from os import mkdir
from os.path import join, exists
import json

# Hyperparameters

Actions = {
    'INCREASE_QUADRATIC': 0,
    'DECREASE_PERCENT': 1,
    'INCREASE_ABSOLUTE': 2,
    'DECREASE_ABSOLUTE': 3,
    'STAY': 4,
    'DECREASE_DRAMATIC': 5,
    'UPDATE_WMAX': 6,
    'RESET_CONGESTION_AVOIDANCE_TIME': 7
}

Rewards = {
    'DROPPED_PACKET': -400,
    'RTT_IS_WAY_TOO_BIG': -1000,
    'DRAMATIC_RTT_INCREASE': -400,
    'INCREASED_RTT': -200,
    'MINOR_RTT_INCREASE': -100,
    'INCREASED_CWND_ABSOLUTE': 10,
    'INCREASED_CWND_PERCENTAGE': 15,
    'NO_REWARD': 0
}

FEATURES = ['rtt', 'dropped_packet']

HYPERPARAMETERS = {
    'Actions': Actions,
    'Rewards': Rewards,
    'DRAMATIC_PERCENT_CHANGE': 0.5,
    'ABSOLUTE_CHANGE': 5,
    'PERCENT_CHANGE': 0.05,
    'FEATURES': FEATURES,
    'RTT_CHANGE_THRESHOLD': 2,
    'RTT_DRAMATIC_CHANGE_THRESHOLD': 4,
    'BATCH_SIZE': 30,
    'REWARD_DECAY': 30,
    'STATE_WINDOW_SIZE': 10,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 20,
    # These constants govern training
    'NUM_EPISODES': 2,
    'TARGET_UPDATE': 15
}

lstm_config = {
    "n_layers": 1,
    "hidden_dim": 15,
    "rdropout": .5, 
    "input_dim": len(FEATURES),
    "output_dim": len(Actions),
    "bidirectional":False,
    "num_layers": 1
}



# directory name, filename
OUTPUT_DIRECTORY = "experiments"
EXPERIMENT_PREFIX = "experiment_"
HYPERPARAMS_FILENAME = "hyperparameters.txt"

mahimahi_settings = {
    'delay': 88,
    'trace_file': '2.64mbps-poisson.trace',
    'queue_type': 'droptail',
    'downlink_queue_options': {
        'bytes': 30000
    }
}

with open(join(OUTPUT_DIRECTORY, HYPERPARAMS_FILENAME), 'w') as hyperparams_file:
    hyperparams_file.write( json.dumps(HYPERPARAMETERS) + "\n" )
    hyperparams_file.write( json.dumps(lstm_config) + "\n" )


def run_experiment(hyperparameters_file_name, experiment_name):
    experiment_dir = join(OUTPUT_DIRECTORY, EXPERIMENT_PREFIX + experiment_name)

    if not exists(OUTPUT_DIRECTORY):
        mkdir(OUTPUT_DIRECTORY)

    if not exists(experiment_dir):
        mkdir(experiment_dir)
    
    hyperparameters = None
    with open(hyperparameters_file_name) as hyperparams_file:
        hyperparameters = json.loads(hyperparams_file.read())

    NUM_EPISODES = hyperparameters['HYPERPARAMETERS']['NUM_EPISODES']
    TARGET_UPDATE = hyperparameters['HYPERPARAMETERS']['TARGET_UPDATE']

    policy_net = LSTM_DQN(hyperparameters['lstm_config'])

    target_net = LSTM_DQN(hyperparameters['lstm_config'])
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.RMSprop(policy_net.parameters())
    transitions = []
    total_losses = []
    for i in range(NUM_EPISODES):
        port = get_open_udp_port()
        strategy = ReinforcementStrategy(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            hyperparameters=hyperparameters['HYPERPARAMETERS'],
            episode_num=i,
            transitions=transitions
        )
        print("***Episode # %d***" % i)
        run_with_mahi_settings(
            mahimahi_settings, 
            10, 
            [Sender(port, strategy)], 
            i % 1 == 0, 
            i, 
            write_to_disk=True, 
            output_dir=OUTPUT_DIRECTORY, 
            experiment_prefix=experiment_dir
        )
        
        total_losses.append(strategy.losses)
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # Print out the loss, using colors to distinguish between
    # episodes.

    colors = ["red", "blue", "green", "yellow", "magenta"]
    start = 0
    for i,loss_array in enumerate(total_losses):
        x = list(range(start, start + len(loss_array)))
        plt.plot(x, loss_array, c=colors[i % 5])
        start += len(loss_array)
    plt.savefig(join(output_dir, experiment_prefix, "loss.png" ))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs an experiment')
    parser.add_argument('--hyperparameters-file', required=True)
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()
    print("Running experiment %s, with hyperparameters file %s" % (args.hyperparameters_file, args.experiment_name))
    run_experiment(args.hyperparameters_file, args.experiment_name)