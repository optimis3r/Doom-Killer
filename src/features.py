class FeaturePreprocessor:
    """
    Extracts scale-invariant features from raw system memory and page allocation metrics
    for OOM prediction.
    """
    def __init__(self):
        self.rel_velocities = []
        self.rel_accelerations = []
        self.prev_pgmajfault = None
        self.majfault_rates = []
        self.cache_to_rss_ratios = []

    def get_features(self, velocity, mem_bytes, limit_bytes, pgmajfault, anon, file):
        if not limit_bytes or limit_bytes == 0:
            limit_bytes = 256 * 1024 * 1024
            
        # Remaining Memory Ratio (headroom)
        rem_mem_ratio = float(limit_bytes - mem_bytes) / float(limit_bytes)
        
        # Relative Velocity: Fraction of limit allocated per second (1 page = 4096 bytes)
        rel_v = float(velocity * 4096) / float(limit_bytes)
        self.rel_velocities.append(rel_v)
        
        # Relative Acceleration
        if len(self.rel_velocities) == 1:
            rel_a = 0.0
        else:
            rel_a = float(rel_v - self.rel_velocities[-2])
        self.rel_accelerations.append(rel_a)
        
        # Major page fault rate (diff per second)
        if self.prev_pgmajfault is None:
            majfault_rate = 0.0
        else:
            majfault_rate = float(max(0, pgmajfault - self.prev_pgmajfault))
        self.prev_pgmajfault = pgmajfault
        self.majfault_rates.append(majfault_rate)

        # Cache-to-RSS ratio (file / anon)
        if anon <= 0:
            cache_to_rss = float(file)
        else:
            cache_to_rss = float(file) / float(anon)
        self.cache_to_rss_ratios.append(cache_to_rss)

        # Keep only the last 3 elements for rolling average window
        if len(self.rel_velocities) > 3:
            self.rel_velocities.pop(0)
        if len(self.rel_accelerations) > 3:
            self.rel_accelerations.pop(0)
        if len(self.majfault_rates) > 3:
            self.majfault_rates.pop(0)
        if len(self.cache_to_rss_ratios) > 3:
            self.cache_to_rss_ratios.pop(0)
            
        # Rolling averages (window=3)
        n = len(self.rel_velocities)
        if n < 3:
            rel_v_roll_3 = float(rel_v)
            rel_a_roll_3 = 0.0
            majfault_rate_roll_3 = float(majfault_rate)
            cache_to_rss_roll_3 = float(cache_to_rss)
        else:
            rel_v_roll_3 = sum(self.rel_velocities) / 3.0
            rel_a_roll_3 = sum(self.rel_accelerations) / 3.0
            majfault_rate_roll_3 = sum(self.majfault_rates) / 3.0
            cache_to_rss_roll_3 = sum(self.cache_to_rss_ratios) / 3.0
            
        return [
            float(rem_mem_ratio),
            float(rel_v),
            float(rel_a),
            float(rel_v_roll_3),
            float(rel_a_roll_3),
            float(majfault_rate),
            float(cache_to_rss),
            float(majfault_rate_roll_3),
            float(cache_to_rss_roll_3)
        ]

