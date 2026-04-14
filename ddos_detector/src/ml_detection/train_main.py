import os
import sys

# Append the root path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sketch.lemon_node import LemonSketch
from sketch.controller import LemonController
from data_prep.csv_to_packets import CSVPacketSimulator
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    print("Please install xgboost and scikit-learn to run the ML training.")
    sys.exit(1)

# Default Sketch config used in USENIX Lemon Paper
LEMON_CONFIG = [
    {"T": 16384, "units": 524288, "bitmap_size": 8},
    {"T": 4096,  "units": 65536,  "bitmap_size": 32},
    {"T": 1024,  "units": 8192,   "bitmap_size": 32},
    {"T": 256,   "units": 2048,   "bitmap_size": 32},
    {"T": 0,     "units": 1024,   "bitmap_size": 512}
]

def main():
    print("=== Hybrid Lemon Sketch + XGBoost Simulator ===")
    
    # 1. Initialize 3 Virtual Switches (Data Plane)
    n_switches = 3
    switches = [LemonSketch(LEMON_CONFIG) for _ in range(n_switches)]
    
    # 2. Initialize Central Controller (Control Plane)
    controller = LemonController(LEMON_CONFIG)
    
    # 3. Simulate Data Injection
    print("[*] Finding all CSV files in MachineLearningCVE...")
    csv_pattern = os.path.join(os.path.dirname(__file__), '../../data/raw/MachineLearningCVE/*.csv')
    all_files = glob.glob(csv_pattern)
    
    if all_files:
        print(f"[*] Found {len(all_files)} files. Loading all data (Warning: Simulating millions of packets will take Time/RAM)...")
        df_list = []
        for file_path in all_files:
            print(f"    - Reading: {os.path.basename(file_path)}")
            # Read everything with latin1 to avoid UnicodeDecodeError in some specific CIC CSVs
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            df.columns = df.columns.str.strip()
            df_list.append(df)
            
        df_epoch = pd.concat(df_list, ignore_index=True)
        print(f"[*] Total Flows combined: {len(df_epoch)}")
    else:
        print("[!] Real CSV not found, falling back to dummy data...")
        dummy_data = {
            'Source IP': [f"192.168.1.{i}" for i in range(1, 100)] + ['10.0.0.1']*50,
            'Destination IP': ['8.8.8.8']*149,
            'Source Port': np.random.randint(1000, 60000, 149),
            'Destination Port': [443]*99 + [53]*50,
            'Protocol': [6]*99 + [17]*50,
            'Total Fwd Packets': np.random.randint(1, 20, 99).tolist() + [5000]*50,
            'Total Backward Packets': np.random.randint(1, 20, 99).tolist() + [5000]*50,
            'Label': ['BENIGN']*99 + ['DNS_Amplification']*50
        }
        df_epoch = pd.DataFrame(dummy_data)
    
    # 4. Run Packets through Data Plane
    simulator = CSVPacketSimulator(n_nodes=n_switches)
    print(f"[*] Simulating dynamic routing of packets to {n_switches} switches...")
    # This acts as our ground truth dictionary {key_flow: label}
    ground_truth = simulator.simulate_epoch(df_epoch, switches)
    
    # 5. Aggregate in Control Plane
    print("[*] Controller aggregating the Lemon Sketches...")
    controller.aggregate_sketches(switches)
    
    # 6. Feature Extraction (Layer 3 Preparation)
    print("[*] Reconstructing Estimated Volumes for XGBoost...")
    features = []
    labels = []
    
    for flow_key, true_label in ground_truth.items():
        est_vol, est_layer = controller.query_volume(flow_key)
        
        # Parse minimal meta-features from flow_key: "src:sport->dst:dport_proto"
        if est_vol == float('inf'):
            est_vol = 9999999 # Handle overflow safely
            
        try:
            proto = int(flow_key.split('_')[-1])
            dport = int(flow_key.split('->')[1].split(':')[1].split('_')[0])
        except:
            proto = 6
            dport = 443

        # Assemble the input vector for XGBoost
        row = {
            'estimated_volume': est_vol,
            'protocol': proto,
            'dest_port': dport,
            # In a full design we would add Entropy and Cardinality here
        }
        features.append(row)
        labels.append(1 if str(true_label).strip().upper() != 'BENIGN' else 0)

    X = pd.DataFrame(features)
    y = np.array(labels)
    
    # 7. Machine Learning Layer 3
    print("\n[*] Training ML Layer 3 (XGBoost)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    try:
        print(classification_report(y_test, preds, target_names=['Benign', 'DDoS Volumetric'], zero_division=0))
    except ValueError:
        print(classification_report(y_test, preds, zero_division=0))

if __name__ == "__main__":
    main()
