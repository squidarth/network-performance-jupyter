import matplotlib
import matplotlib.pyplot as plt
import re
import os
from subprocess import Popen
import socket
from threading import Thread
from typing import Dict, List
from src.senders import Sender
from os.path import join


RECEIVER_FILE = "run_receiver.py"
AVERAGE_SEGMENT_SIZE = 80
QUEUE_LOG_FILE = "downlink_queue.log"
QUEUE_LOG_TMP_FILE = "downlink_queue_tmp.log"

DROP_LOG = "debug_log.log"
DROP_LOG_TMP_FILE = "debug_log_tmp.log"

def generate_mahimahi_command(mahimahi_settings: Dict) -> str:
    if mahimahi_settings.get('loss'):
        loss_directive = "mm-loss downlink %f" % mahimahi_settings.get('loss')
    else:
        loss_directive = ""

    queue_type =  mahimahi_settings.get('queue_type', 'droptail')

    if mahimahi_settings.get('downlink_queue_options'):
        downlink_queue_options = "--downlink-queue-args=" + ",".join(
             ["%s=%s" % (key, value)
             for key, value in mahimahi_settings.get('downlink_queue_options').items()]
        )
    else:
        downlink_queue_options = ""

    if mahimahi_settings.get('uplink_queue_options'):
        uplink_queue_options = " ".join(
            ["--downlink-queue-args=%s=%s" % (key, value)
             for key, value in mahimahi_settings.get('uplink_queue_options').items()]
        )
    else:
        uplink_queue_options = ""

    return "mm-delay {delay} {loss_directive} mm-link traces/{trace_file} traces/{trace_file} --downlink-queue={queue_type} {downlink_queue_options} {uplink_queue_options} --downlink-queue-log={queue_log_file}".format(
      delay=mahimahi_settings['delay'],
      downlink_queue_options=downlink_queue_options,
      uplink_queue_options=uplink_queue_options,
      loss_directive=loss_directive,
      trace_file=mahimahi_settings['trace_file'],
      queue_type=queue_type,
      queue_log_file=QUEUE_LOG_FILE
    )

def get_open_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

SENDER_COLORS = ["blue", "red", "green", "cyan", "magenta", "yellow", "black"]

def print_performance(
        senders: List[Sender],
        num_seconds: int,
        episode_num : int,
        write_to_disk : bool ,
        output_dir : str,
        experiment_dir : str
        ):

    if write_to_disk:
        with open(join(experiment_dir,  "episode_" + str(episode_num) + "_stats.txt" ), 'w') as out_stats:
            for sender in senders:
                out_stats.write("Results for sender %d, with strategy: %s" % (sender.port, sender.strategy.__class__.__name__) + "\n")
                out_stats.write("**Throughput:**                           %f bytes/s" % (AVERAGE_SEGMENT_SIZE * (sender.strategy.ack_count/num_seconds)) + "\n")
                out_stats.write("**Average RTT:**                          %f ms" % ((float(sum(sender.strategy.rtts))/len(sender.strategy.rtts)) * 1000) + "\n")
                out_stats.write("\n")
    else:
        for sender in senders:
            print("Results for sender %d, with strategy: %s" % (sender.port, sender.strategy.__class__.__name__))
            print("**Throughput:**                           %f bytes/s" % (AVERAGE_SEGMENT_SIZE * (sender.strategy.ack_count/num_seconds)))
            print("**Average RTT:**                          %f ms" % ((float(sum(sender.strategy.rtts))/len(sender.strategy.rtts)) * 1000))
            print("")


    # Compute the queue log stuff
    queue_log_lines = open(QUEUE_LOG_TMP_FILE).read().split("\n")[1:]
    regex = re.compile("\d+ # (\d+)")

    plt.plot([int(regex.match(line).group(1)) for line in queue_log_lines if regex.match(line) is not None])

    plt.xlabel("Time")
    plt.ylabel("Link Queue Size")

    if write_to_disk:
        plt.savefig(join(experiment_dir, "episode_" + str(episode_num) + "_link-queue-size.png" ))
        plt.close()
    else:
        plt.show()

    handles = []
    for idx, sender in enumerate(senders):
        plt.plot(*zip(*sender.strategy.cwnds), c=SENDER_COLORS[idx], label=sender.strategy.__class__.__name__)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Congestion Window Size")

    if write_to_disk:
        plt.savefig(join(experiment_dir, "episode_" + str(episode_num) + "_cwnd.png" ))
        plt.close()
    else:
        plt.show()
        print("")

    for idx, sender in enumerate(senders):
        plt.plot(*zip(*sender.strategy.rtt_recordings), c=SENDER_COLORS[idx], label=sender.strategy.__class__.__name__)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Current RTT")
    if write_to_disk:
        plt.savefig(join(experiment_dir, "episode_" + str(episode_num) +"_rtt.png" ))
        plt.close()
    else:
        plt.show()

def run_with_mahi_settings(
        mahimahi_settings: Dict, 
        seconds_to_run: int, 
        senders: List, 
        should_print_performance: bool , 
        episode_num : int,
        write_to_disk : bool ,
        output_dir : str,
        experiment_dir : str
        ):
    mahimahi_cmd = generate_mahimahi_command(mahimahi_settings)

    sender_ports = " ".join(["$MAHIMAHI_BASE %s" % sender.port for sender in senders])

    cmd = "%s -- sh -c 'python3 %s %d %s' > out.out" % (mahimahi_cmd, RECEIVER_FILE, seconds_to_run, sender_ports)
    Popen(cmd, shell=True)
    for sender in senders:
        sender.handshake()
    threads = [Thread(target=sender.run, args=[seconds_to_run]) for sender in senders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    os.rename(QUEUE_LOG_FILE, QUEUE_LOG_TMP_FILE)
    #os.rename(DROP_LOG, DROP_LOG_TMP_FILE)

    if should_print_performance:
        print_performance(senders, seconds_to_run, episode_num, write_to_disk, output_dir, experiment_dir)
    Popen("pkill -f mm-link", shell=True).wait()
    Popen("pkill -f run_receiver", shell=True).wait()
