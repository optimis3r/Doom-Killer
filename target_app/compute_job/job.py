import numpy as np
import time
import random
import sys

def main():
    print("Compute job started...", flush=True)
    while True:
        try:
            # Random size to vary memory usage dynamically but bounded
            size = random.randint(500, 2000)
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            c = np.dot(a, b)
            del a, b, c
            time.sleep(random.uniform(0.1, 0.5))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)
            time.sleep(1)

if __name__ == "__main__":
    main()
