class Sender(object):
    def __init__(self, port: int) -> None:
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)
        self.poller.modify(self.sock, ALL_FLAGS)
        self.peer_addr = None
        
        # Congestion control variables
        self.seq_num = 0
        self.next_ack = 0

        # Variables for the reinforment learning
        self.sent_bytes = 0

        self.min_rtt = float('inf')
        
        self.unacknowledged_packets = {}
        self.rtts = []
        self.start_time = time.time()
        self.total_acks = 0
        self.num_duplicate_acks = 0
        self.curr_duplicate_acks = 0
        self.cwnds = []
        # List of tuples of ack_time and seq_num
        self.times_of_acknowledgements = []
        
        
        if self.cwnd == None:
            raise "No initial setting for cwnd"
        
    def window_is_open(self):
        # Returns true if the congestion window is not full
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        raise NotImplementedError
    
    def recv(self):
        raise NotImplementedError


    def handshake(self):
        """Handshake to establish connection with receiver."""

        while True:
            msg, addr = self.sock.recvfrom(1600)
            parsed_handshake = json.loads(msg.decode())
            if parsed_handshake.get('handshake') and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto(json.dumps({'handshake': True}).encode(), self.peer_addr)
                print('[sender] Connected to receiver: %s:%s\n' % addr)
                break
        self.sock.setblocking(0)

    def run(self, seconds_to_run: int):
        curr_flags = ALL_FLAGS
        TIMEOUT = 1000  # ms
        start_time = time.time()

        while time.time() - start_time < seconds_to_run:

            events = self.poller.poll(TIMEOUT)
            if not events:
                self.send()
            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:
                    self.recv()

                if flag & WRITE_FLAGS:
                    self.send()


class FixedWindowSender(Sender):
    def __init__(self, cwnd: int, port: int) -> None:
        self.cwnd = cwnd
                
        super().__init__(port)


    def send(self):
        if not self.window_is_open():
            return
        
        serialized_data = json.dumps({
            'seq_num': self.seq_num,
            'send_ts': time.time(),
            'sent_bytes': self.sent_bytes
        })
        self.unacknowledged_packets[self.seq_num] = True
        self.seq_num += 1
        self.sock.sendto(serialized_data.encode(), self.peer_addr)
        time.sleep(0)
    
    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1600)

        ack = json.loads(serialized_ack.decode())
        if ack.get('handshake'):
            return
        
        self.total_acks += 1
        self.times_of_acknowledgements.append(((time.time() - self.start_time), ack['seq_num']))
        if self.unacknowledged_packets.get(ack['seq_num']) is None:
            # Duplicate ack
            self.num_duplicate_acks += 1
            self.curr_duplicate_acks += 1
            
            if self.curr_duplicate_acks == 3:
                # Received 3 duplicate acks, retransmit
                self.curr_duplicate_acks = 0
                self.seq_num = ack['seq_num'] + 1
        else:
            del self.unacknowledged_packets[ack['seq_num']]
            self.next_ack = max(self.next_ack, ack['seq_num'] + 1)
            self.sent_bytes += ack['ack_bytes']
            rtt = float(time.time() - ack['send_ts'])
            self.rtts.append(rtt)
        self.cwnds.append(self.cwnd) 



class TahoeSender(Sender):
    def __init__(self, slow_start_thresh: int, initial_cwnd: int, port: int) -> None:
        self.initial_cwnd = initial_cwnd
        self.cwnd = initial_cwnd
        self.slow_start_thresh = slow_start_thresh
        self.fast_retransmit_packet = None
        self.retransmitting_packet = False
        
        self.duplicated_ack = None

        super().__init__(port)
        
    def window_is_open(self) -> bool:
        # Returns true if the congestion window is not full
       # print(len(self.unacknowledged_packets))
        #return len(self.unacknowledged_packets) < self.cwnd
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        if self.fast_retransmit_packet and not self.retransmitting_packet:
            print("Retransmitting seq number %d" % self.fast_retransmit_packet['seq_num'])
            data = self.fast_retransmit_packet
            serialized_data = json.dumps(data)
            self.retransmitting_packet = True

        elif self.window_is_open():
            data = {
                'seq_num': self.seq_num,
                'send_ts': time.time(),
                'sent_bytes': self.sent_bytes
            }
            
            self.unacknowledged_packets[self.seq_num] = data
            self.seq_num += 1
        else:
            return
    
        serialized_data = json.dumps(data)
        self.sock.sendto(serialized_data.encode(), self.peer_addr)
        time.sleep(0)
    
    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1600)

        ack = json.loads(serialized_ack.decode())
        if ack.get('handshake'):
            return
        
        self.total_acks += 1
        self.times_of_acknowledgements.append(((time.time() - self.start_time), ack['seq_num']))
        #print("Next ack %d" % self.next_ack)
        #print("Ack: %d" %ack['seq_num'] )

        
        if self.unacknowledged_packets.get(ack['seq_num']) is None:
            # Duplicate ack

            self.num_duplicate_acks += 1
            if self.duplicated_ack and ack['seq_num'] == self.duplicated_ack['seq_num']:
                self.curr_duplicate_acks += 1
            else:
                self.duplicated_ack = ack
                self.curr_duplicate_acks = 1
            
            if self.curr_duplicate_acks == 3:
                # Received 3 duplicate acks, retransmit
                
                # Go into fast retransmit

                self.fast_retransmit_packet = self.unacknowledged_packets[ack['seq_num'] + 1]
                self.slow_start_thresh = self.cwnd/2
                self.cwnd = 1
        elif ack['seq_num'] >= self.next_ack:
            if self.fast_retransmit_packet:
                print("Leaving fast retransmit on seq_num %d" % ack['seq_num'])
                self.fast_retransmit_packet = None
                self.retransmitting_packet = False
                self.curr_duplicate_acks = 0
                self.seq_num = ack['seq_num'] + 1
                print("seq_num is now %d" % self.seq_num)
                # Acknowledge all packets where seq_num < ack['seq_num']
                # Throw out all unacknowledged packets and start over
            
            # Acknowledge all packets where seq_num < ack['seq_num']
            self.unacknowledged_packets = {
                k:v
                for k,v in
                self.unacknowledged_packets.items()
                if k > ack['seq_num']
            }
            self.next_ack = max(self.next_ack, ack['seq_num'] + 1)
            self.sent_bytes += ack['ack_bytes']
            rtt = float(time.time() - ack['send_ts'])
            self.rtts.append(rtt)
            if self.cwnd < self.slow_start_thresh:
                # In slow start
                self.cwnd += 1 
            elif ack['seq_num'] % self.cwnd == 0:
                # In congestion avoidance
                self.cwnd += 1
    
        self.cwnds.append(self.cwnd) 
