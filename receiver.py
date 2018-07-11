import sys
import json
import socket
import select

READ_FLAGS = select.POLLIN | select.POLLPRI
WRITE_FLAGS = select.POLLOUT
ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS


class Receiver(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

    def cleanup(self):
        self.sock.close()

    def construct_ack_from_data(self, serialized_data):
        """Construct a serialized ACK that acks a serialized datagram."""

        data = json.loads(serialized_data)
        print(serialized_data)

        return json.dumps({
          'seq_num': data['seq_num'],
          'send_ts': data['send_ts'],
          'sent_bytes': data['sent_bytes'],
          'delivered_time': data['delivered_time'],
          'delivered': data['delivered'],
          'ack_bytes': len(serialized_data)
        })


    def handshake(self):
        """Handshake with peer sender. Must be called before run()."""

        #self.sock.setblocking(0)  # non-blocking UDP socket

        TIMEOUT = 1000  # ms

        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)

        while True:
            self.sock.sendto('Hello from receiver', self.peer_addr)
            events = self.poller.poll(TIMEOUT)
            print(events)

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
                    print(type(addr[0]))
                    parsed_addr = (addr[0].decode(), addr[1])
                    print(msg)
                    print(addr)

                    if addr == self.peer_addr:
                        if msg.decode() != 'Hello from sender':
                            # 'Hello from sender' was presumably lost
                            # received subsequent data from peer sender
                            ack = self.construct_ack_from_data(msg)
                            if ack is not None:
                                self.sock.sendto(ack, self.peer_addr)
                        return

    def run(self):
        self.sock.setblocking(1)  # blocking UDP socket

        while True:
            serialized_data, addr = self.sock.recvfrom(1600)

            if addr == self.peer_addr:
                ack = self.construct_ack_from_data(serialized_data)
                if ack is not None:
                    self.sock.sendto(ack, self.peer_addr)
