#!/usr/bin/env python

import argparse
from src.receiver import Receiver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('running_time')
    parser.add_argument('ip_port_pairs', nargs='*')
    args = parser.parse_args()
    running_time = args.running_time
    peers = args.ip_port_pairs

    receiver = Receiver(int(running_time), [(peers[i], int(peers[i+1])) for i in range(0, len(peers), 2)])

    try:
        receiver.perform_handshakes()
        receiver.run()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.cleanup()


if __name__ == '__main__':
    main()
