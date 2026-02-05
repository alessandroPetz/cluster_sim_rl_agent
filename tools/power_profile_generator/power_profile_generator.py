import random
import os

# Parametri iniziali
MIN_VAL = 740
MAX_VAL = 850
N = 4000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "./power_4000jobs.txt")

with open(OUTPUT_FILE, "w") as f:
    for i in range(N + 1):  # da 0 a N
        value = random.randint(MIN_VAL, MAX_VAL)
        f.write(f"{i} {value} CPU\n")

print(f"File generato: {OUTPUT_FILE}")