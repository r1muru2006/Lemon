import numpy as np
import math

class LemonController:
    """
    Simulation of the Centralized Control Plane that aggregates 
    sketches from multiple switches.
    """
    def __init__(self, layer_configs, max_hash_val=0xFFFFFFFF):
        self.layer_configs = layer_configs
        self.n_layers = len(layer_configs)
        self.max_hash_val = max_hash_val
        
        self.global_layers = []
        for config in layer_configs:
            layer_mem = np.zeros((config["units"], config["bitmap_size"]), dtype=np.uint8)
            self.global_layers.append(layer_mem)

    def aggregate_sketches(self, sketch_list):
        """
        Merge multiple LemonSketches into a network-wide sketch using OR operation.
        """
        for sketch in sketch_list:
            raw_layers = sketch.get_raw_sketch()
            for i in range(self.n_layers):
                # Bitwise OR aggregation across multiple physical switches
                self.global_layers[i] = np.bitwise_or(self.global_layers[i], raw_layers[i])

    def _hash(self, key, seed):
        import hashlib
        h = hashlib.md5(f"{seed}{key}".encode('utf-8')).hexdigest()
        return int(h[:8], 16)

    def query_volume(self, key_flow):
        """
        Algorithm 2: Lemon sketch query for flow size
        Calculates the volume of a flow using the best sample layer in the global sketch.
        """
        h_slot = self._hash(key_flow, seed="slot")
        
        bitmaps = []
        for i in range(self.n_layers):
            index_slot = h_slot % self.layer_configs[i]["units"]
            bitmap = self.global_layers[i][index_slot]
            bitmaps.append(bitmap)
            
        Tz = 0.2
        T_prev = self.max_hash_val
        
        # Bottom-up search for best sample layer
        for i in range(self.n_layers):
            T_i = self.layer_configs[i]["T"]
            Sb = self.layer_configs[i]["bitmap_size"]
            
            # Count zeros in the bitmap
            Zi = np.sum(bitmaps[i] == 0)
            
            if Zi >= Tz * Sb:
                # Estimate total volume based on this layer's sampling rate
                if Zi == 0: # prevent log(0) just in case
                    return float('inf'), None
                
                E = -Sb * math.log(Zi / Sb)
                # Correct by sampling rate
                sample_rate = (T_prev - T_i) / self.max_hash_val
                if sample_rate > 0:
                    E = E / sample_rate
                
                return E, i # return Estimated volume and Layer used
                
            T_prev = T_i
            
        return float('inf'), None # Overflow

    def reset(self):
        """Reset the global sketch for a new epoch window"""
        for i in range(self.n_layers):
            self.global_layers[i].fill(0)
