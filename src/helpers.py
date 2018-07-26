import matplotlib.pyplot as plt
import re
import os
from subprocess import Popen
import socket
from threading import Thread
from typing import Dict, List
from src.senders import Sender

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


def print_performance(sender: Sender, num_seconds: int):
    print("Results for sender %d:" % sender.port)
    print("Total Acks: %d" % sender.strategy.total_acks)
    print("Num Duplicate Acks: %d" % sender.strategy.num_duplicate_acks)

    print("%% duplicate acks: %f" % ((float(sender.strategy.num_duplicate_acks * 100))/sender.strategy.total_acks))
    print("Throughput (bytes/s): %f" % (AVERAGE_SEGMENT_SIZE * (sender.strategy.ack_count/num_seconds)))
    print("Average RTT (ms): %f" % ((float(sum(sender.strategy.rtts))/len(sender.strategy.rtts)) * 1000))


    # Compute the queue log stuff
    queue_log_lines = open(QUEUE_LOG_TMP_FILE).read().split("\n")[1:]
    regex = re.compile("\d+ # (\d+)")

    plt.plot([int(regex.match(line).group(1)) for line in queue_log_lines if regex.match(line) is not None])

    plt.xlabel("Time")
    plt.ylabel("Link Queue Size")
    plt.show()
    print(" ")

    timestamps = [ ack[0] for ack in sender.strategy.times_of_acknowledgements]
    seq_nums = [ ack[1] for ack in sender.strategy.times_of_acknowledgements]

    plt.scatter(timestamps, seq_nums)
    plt.xlabel("Timestamps")
    plt.ylabel("Sequence Numbers")

    plt.show()

    plt.plot(sender.strategy.cwnds)
    plt.xlabel("Time")
    plt.ylabel("Congestion Window Size")
    plt.show()
    print("")

    plt.plot(sender.strategy.rtts)
    plt.xlabel("Time")
    plt.ylabel("Current RTT")
    plt.show()
    print("")

    if len(sender.strategy.slow_start_thresholds) > 0:
        plt.plot(sender.strategy.slow_start_thresholds)
        plt.xlabel("Time")
        plt.ylabel("Slow start threshold")
        plt.show()
    print("")

def run_with_mahi_settings(mahimahi_settings: Dict, seconds_to_run: int, senders: List):
    mahimahi_cmd = generate_mahimahi_command(mahimahi_settings)

    sender_ports = " ".join(["$MAHIMAHI_BASE %s" % sender.port for sender in senders])

    cmd = "%s -- sh -c 'python3 %s %d %s' > out.out" % (mahimahi_cmd, RECEIVER_FILE, seconds_to_run, sender_ports)
    receiver_process = Popen(cmd, shell=True)
    for sender in senders:
        sender.handshake()
    threads = [Thread(target=sender.run, args=[seconds_to_run]) for sender in senders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    os.rename(QUEUE_LOG_FILE, QUEUE_LOG_TMP_FILE)
    #os.rename(DROP_LOG, DROP_LOG_TMP_FILE)

    for sender in senders:
        print_performance(sender, seconds_to_run)
    receiver_process.kill()
