import numpy as np
import hashlib

class LemonSketch:
    """
    Simulation of the Lemon Sketch data plane as described in the 
    USENIX Security 25 paper.
    """
    def __init__(self, layer_configs, max_hash_val=0xFFFFFFFF):
        """
        layer_configs: list of dicts. Ex:
        [
            {"T": 16384, "units": 524288, "bitmap_size": 8},
            {"T": 4096,  "units": 65536,  "bitmap_size": 32},
            {"T": 1024,  "units": 8192,   "bitmap_size": 32},
            {"T": 256,   "units": 2048,   "bitmap_size": 32},
            {"T": 0,     "units": 1024,   "bitmap_size": 512}
        ]
        """
        self.layer_configs = layer_configs
        self.n_layers = len(layer_configs)
        self.max_hash_val = max_hash_val
        self.Th = 100 # Threshold for Heavy Hitter tracking
        
        # Initialize the arrays representing hardware memory on the switch
        self.layers = []
        for config in layer_configs:
            # Each unit has `bitmap_size` bits. We use uint8 arrays to simulate bits (0/1).
            # True hardware uses raw bit arrays. Here numpy uint8 is fast enough for simulation.
            layer_mem = np.zeros((config["units"], config["bitmap_size"]), dtype=np.uint8)
            self.layers.append(layer_mem)
            
        self.heavy_hitters = {}

    def _hash(self, key, seed):
        """Deterministic hashing simulation for Hd, Hs, Hb"""
        h = hashlib.md5(f"{seed}{key}".encode('utf-8')).hexdigest()
        return int(h[:8], 16) # use first 8 hex chars (32 bits)
        
    def update(self, key_flow, key_pkg):
        """
        Algorithm 1: Lemon sketch updating
        key_flow: e.g. "192.168.1.1:80->10.0.0.1:443_TCP"
        key_pkg: unique per packet e.g. "IP_ID=1234_TCP_SEQ=5678"
        """
        h_slot = self._hash(key_flow, seed="slot")
        h_layer = self._hash(key_pkg, seed="layer") % self.max_hash_val
        h_bitmap = self._hash(key_pkg, seed="bitmap")
        
        T_prev = self.max_hash_val
        
        # Select layer based on packet hash (h_layer)
        for i in range(self.n_layers):
            T_i = self.layer_configs[i]["T"]
            
            # check if h_layer falls in [T_i, T_{i-1})
            if T_i <= h_layer < T_prev:
                # Find which slot to update
                n_units = self.layer_configs[i]["units"]
                index_slot = h_slot % n_units
                
                # Find which bit to update
                bitmap_size = self.layer_configs[i]["bitmap_size"]
                index_bitmap = h_bitmap % bitmap_size
                
                # Set the bit to 1
                if self.layers[i][index_slot, index_bitmap] == 0:
                    self.layers[i][index_slot, index_bitmap] = 1
            
            T_prev = T_i
            
        # Heavy Hitter explicit tracking (optional but in paper)
        if h_layer <= self.Th:
            self.heavy_hitters[key_flow] = True
            
    def get_raw_sketch(self):
        """Returns the internal raw memory structure to send to Controller"""
        return self.layers
