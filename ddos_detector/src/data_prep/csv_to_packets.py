import pandas as pd
import random
import uuid

class CSVPacketSimulator:
    """
    Simulates packet streams from CICDDoS CSV files to avoid the bottleneck
    of parsing raw PCAP files in Python.
    """
    def __init__(self, n_nodes=3):
        self.n_nodes = n_nodes

    def simulate_epoch(self, df_chunk, lemon_nodes):
        """
        Takes a chunk of flows (pandas dataframe representing an epoch of time)
        and simulates the packet routing to distributed lemon nodes.
        """
        # Ground truth mapping for evaluating ML later
        ground_truth = {}
        print(f"[*] Starting Flow Injection for {len(df_chunk)} flows...")
        from tqdm import tqdm
        for index, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc="Simulating Distributed Packets"):
            try:
                src_ip = row.get('Source IP', f"src_{index}")
                dst_ip = row.get('Destination IP', f"dst_{index}")
                src_port = row.get('Source Port', 0)
                dst_port = row.get('Destination Port', 0)
                proto = row.get('Protocol', 6) # tcp
                label = row.get('Label', 'BENIGN')
                
                # Handling different CICDDoS column names variants
                if 'Total Fwd Packets' in row:
                    fwd_pkts = int(row['Total Fwd Packets'])
                    bwd_pkts = int(row['Total Backward Packets'])
                    total_pkts = fwd_pkts + bwd_pkts
                elif 'Tot Fwd Pkts' in row:
                    total_pkts = int(row['Tot Fwd Pkts']) + int(row['Tot Bwd Pkts'])
                else:
                    total_pkts = 10 # default fallback
                    
            except Exception as e:
                continue

            key_flow = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}_{proto}"
            ground_truth[key_flow] = label
            
            # Simulate individual packets and split them across nodes (Dynamic Routing)
            for pkt_idx in range(total_pkts):
                # Unique packet identifier. In reality this is a hash of IP/TCP headers + sequence.
                key_pkg = f"{key_flow}_seq_{pkt_idx}_{uuid.uuid4().hex[:4]}"
                
                # Randomly assign packet to one of the Virtual Switches
                target_node = random.choice(lemon_nodes)
                target_node.update(key_flow, key_pkg)
                
        return ground_truth
