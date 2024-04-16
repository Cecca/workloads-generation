import joblib
import os

DATA_DIR = os.environ.get("WORKGEN_DATA_DIR", ".data") # on nefeli: export WORKGEN_DATA_DIR=/mnt/hddhelp/workgen_data/
MEM = joblib.Memory(os.path.join(DATA_DIR, "joblib-cache"))
