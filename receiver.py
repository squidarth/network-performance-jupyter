import sys
import json
import socket
import select
from typing import List, Dict

READ_FLAGS = select.POLLIN | select.POLLPRI
WRITE_FLAGS = select.POLLOUT
ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS

class Peer(object):
    def __init__(self, port):
        self.port = port
        self.seq_num = -1
        self.attempts = 0
        self.previous_ack = None

    def update(self, ack: Dict):
        self.previous_ack = ack
        self.attempts = 0
        self.seq_num = ack['seq_num']

class Receiver(object):
    def __init__(self, peers: List):
        self.peers = {}
        for peer in peers:
            self.peers[peer] = Peer(peer[1])

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

    def cleanup(self):
        self.sock.close()

    def construct_ack_from_data(self, serialized_data: str):
        """Construct a serialized ACK that acks a serialized datagram."""

        data = json.loads(serialized_data)

        return json.dumps({
          'seq_num': data['seq_num'],
          'send_ts': data['send_ts'],
          'sent_bytes': data['sent_bytes'],
          'ack_bytes': len(serialized_data)
        })


    def perform_handshakes(self):
        """Handshake with peer sender. Must be called before run()."""

        self.sock.setblocking(0)  # non-blocking UDP socket

        TIMEOUT = 1000  # ms

        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)
        # Copy self.peers
        unconnected_peers = self.peers.keys()

        while len(unconnected_peers) > 0:
            for peer in unconnected_peers:
                self.sock.sendto(json.dumps({'handshake': True}).encode(), peer)

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write(
                        '[receiver] Handshake failed after 10 retries\n')
                    return
                else:
                    sys.stderr.write(
                        '[receiver] Handshake timed out and retrying...\n')
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1600)

                    if addr in unconnected_peers:
                        if json.loads(msg.decode()).get('handshake'):
                            unconnected_peers.remove(addr)

    def next_ack(self, serialized_data: str):
        """Construct a serialized ACK that acks a serialized datagram."""
        data = json.loads(serialized_data)
        return {
          'seq_num': data['seq_num'],
          'send_ts': data['send_ts'],
          'sent_bytes': data['sent_bytes'],
          'ack_bytes': len(serialized_data)
        }

    def perform_handshakes(self):
        """Handshake with peer sender. Must be called before run()."""

        self.sock.setblocking(0)  # non-blocking UDP socket

        TIMEOUT = 1000  # ms

        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)
        # Copy self.peers
        unconnected_peers = list(self.peers.keys())

        while len(unconnected_peers) > 0:
            for peer in unconnected_peers:
                self.sock.sendto(json.dumps({'handshake': True}).encode(), peer)

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write(
                        '[receiver] Handshake failed after 10 retries\n')
                    return
                else:
                    sys.stderr.write(
                        '[receiver] Handshake timed out and retrying...\n')
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1600)

                    if addr in unconnected_peers:
                        if json.loads(msg.decode()).get('handshake'):
                            unconnected_peers.remove(addr)

    def run(self):
        self.sock.setblocking(1)  # blocking UDP socket

        while True:
            serialized_data, addr = self.sock.recvfrom(1600)

            if addr in self.peers:
                peer = self.peers[addr]


                data = json.loads(serialized_data)
                seq_num = data['seq_num']

                ack = None
                if seq_num == peer.seq_num + 1:
                    ack = self.next_ack(serialized_data)
                    peer.update(ack)
                elif seq_num > peer.seq_num + 1 and peer.attempts < 3:
                    ack = peer.previous_ack
                    peer.attempts += 1

                if ack is not None:
                    self.sock.sendto(json.dumps(ack).encode(), addr)
