import argparse
import numpy as np

def generate_trace_file(trace_file_path: str, bandwidth: float):
    num_packets = int(float(bandwidth) * 5000)
    timestamp_list = np.linspace(0, 60000, num=num_packets, endpoint=False)
    with open(trace_file_path, 'w') as trace:
        for ts in timestamp_list:
            trace.write('%d\n' % ts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bandwidth', metavar='Mbps', required=True,
                        help='constant bandwidth (Mbps)')
    parser.add_argument('--trace-path', metavar='TRACE_PATH', required=True,
                        help='file to output trace')

    args = parser.parse_args()
    generate_trace_file(args.trace_path, float(args.bandwidth))
