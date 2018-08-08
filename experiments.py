import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
# Use this if you are writing to disk rather than
# using ipython.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.helpers import run_with_mahi_settings, get_open_udp_port
from src.senders import Sender
from src.ml_strategy import ReinforcementStrategy
from src.ml_helpers import LSTM_DQN

from os import mkdir
from os.path import join, exists
import json

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

def run_experiment(hyperparameters_file_name, experiment_name):
    experiment_dir = join(OUTPUT_DIRECTORY, EXPERIMENT_PREFIX + experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not exists(OUTPUT_DIRECTORY):
        mkdir(OUTPUT_DIRECTORY)

    if not exists(experiment_dir):
        mkdir(experiment_dir)

    hyperparameters = None
    with open(hyperparameters_file_name) as hyperparams_file:
        hyperparameters = json.loads(hyperparams_file.read())

    NUM_EPISODES = hyperparameters['HYPERPARAMETERS']['NUM_EPISODES']
    TARGET_UPDATE = hyperparameters['HYPERPARAMETERS']['TARGET_UPDATE']

    # policy_net = LSTM_DQN(hyperparameters['lstm_config'], device, use_cuda=torch.cuda.is_available() )
    # target_net = LSTM_DQN(hyperparameters['lstm_config'], device, use_cuda=torch.cuda.is_available() )
    policy_net = LSTM_DQN(hyperparameters['lstm_config'], device).to(device=device)
    target_net = LSTM_DQN(hyperparameters['lstm_config'], device).to(device=device)

    # if torch.cuda.is_available():
    #     policy_net.cuda()
    #     target_net.cuda()

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.RMSprop(policy_net.parameters())
    transitions = []
    total_losses = []
    for i in range(NUM_EPISODES):
        port = get_open_udp_port()
        strategy = ReinforcementStrategy(
            policy_net=policy_net,
            target_net=target_net,
            device=device,
            optimizer=optimizer,
            hyperparameters=hyperparameters['HYPERPARAMETERS'],
            episode_num=i,
            transitions=transitions
        )
        print("***Episode # %d***" % i)
        run_with_mahi_settings(
            mahimahi_settings,
            60,
            [Sender(port, strategy)],
            True,       # always log rtt, cwnd, stats, and queue usage
            i,
            write_to_disk=True,
            output_dir=OUTPUT_DIRECTORY,
            experiment_dir=experiment_dir
        )
        total_losses.append(strategy.losses)
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # model persistence here
            policy_net_filename = join( experiment_dir, "policy-net_episode-" + str(i) + ".model" )
            torch.save(policy_net.state_dict(), policy_net_filename)

    # Print out the loss, using colors to distinguish between
    # episodes.

    colors = ["red", "blue", "green", "yellow", "magenta"]
    start = 0
    for i,loss_array in enumerate(total_losses):
        x = list(range(start, start + len(loss_array)))
        plt.plot(x, loss_array, c=colors[i % 5])
        start += len(loss_array)
    plt.savefig(join(experiment_dir, "loss.png" ))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs an experiment')
    parser.add_argument('--hyperparameters-file', required=True)
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()
    print("Running experiment %s, with hyperparameters file %s" % (args.hyperparameters_file, args.experiment_name))
    run_experiment(args.hyperparameters_file, args.experiment_name)
