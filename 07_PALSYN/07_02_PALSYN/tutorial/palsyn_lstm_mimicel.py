import os
from pathlib import Path

import pm4py

from PALSYN.postprocessing.log_postprocessing import clean_xes_file
from PALSYN.synthesizer import LSTMSynthesizer


# ============================================================
# CONFIG
# ============================================================

SEED = 88
EPSILON = 15.0
SAMPLE_SIZE = 1285

LOG_PATH = Path("PALSYN/data/MIMICEL/mimicel_train.xes")

MODEL_DIR = Path("PALSYN/models/mimicel_train/lstm_seed88_eps15")
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "best.keras"

RESULT_DIR = Path("PALSYN/results/mimicel_train")
XES_OUT = RESULT_DIR / "palsyn_synthetic_lstm_seed88_eps15.xes"
CSV_OUT = RESULT_DIR / "palsyn_synthetic_lstm_seed88_eps15.csv"
XLSX_OUT = RESULT_DIR / "palsyn_synthetic_lstm_seed88_eps15.xlsx"


# ============================================================
# SETUP
# ============================================================

os.makedirs(CHECKPOINT_PATH.parent, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("LOG_PATH:", LOG_PATH)
print("LOG exists:", LOG_PATH.exists())
print("MODEL_DIR:", MODEL_DIR)
print("RESULT_DIR:", RESULT_DIR)

if not LOG_PATH.exists():
    raise FileNotFoundError(f"XES file not found: {LOG_PATH}")


# ============================================================
# LOAD EVENT LOG
# ============================================================

event_log = pm4py.read_xes(str(LOG_PATH))

print("trace count:", len(event_log))

activities = sorted(set(
    event["concept:name"]
    for trace in event_log
    for event in trace
))

print("activity count:", len(activities))
print("activities:", activities)


# ============================================================
# TRAIN PALSYN LSTM
# ============================================================

synthesizer = LSTMSynthesizer(
    pre_processing={
        "max_clusters": 10,
        "trace_quantile": 0.9,
        "seed": SEED,
    },
    model={
        "embedding_output_dims": 64,
        "epochs": 5,
        "batch_size": 64,
        "validation_split": 0.15,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "units_per_layer": [32, 16],
        "dropout": 0.0,
        "bidirectional": True,
    },
    dp_optimizer={
        "epsilon": EPSILON,
        "learning_rate": 5e-4,
        "l2_norm_clip": 1.0,
    },
)

synthesizer.fit(event_log)


# ============================================================
# SAVE MODEL
# ============================================================

synthesizer.save_model(str(MODEL_DIR))
print("saved model:", MODEL_DIR)


# ============================================================
# LOAD MODEL AND SAMPLE
# ============================================================

loaded_model = LSTMSynthesizer.load(str(MODEL_DIR))

synthetic_df = loaded_model.sample(
    sample_size=SAMPLE_SIZE,
    batch_size=64,
)

print("synthetic sample generated")


# ============================================================
# SAVE SYNTHETIC LOG
# ============================================================

synthetic_event_log = pm4py.convert_to_event_log(synthetic_df)

pm4py.write_xes(synthetic_event_log, str(XES_OUT))
clean_xes_file(str(XES_OUT), str(XES_OUT))

synthetic_event_df = pm4py.convert_to_dataframe(synthetic_event_log)

if "time:timestamp" in synthetic_event_df.columns:
    synthetic_event_df["time:timestamp"] = synthetic_event_df["time:timestamp"].astype(str)

synthetic_event_df.to_csv(CSV_OUT, index=False)
synthetic_event_df.to_excel(XLSX_OUT, index=False)

print("saved xes:", XES_OUT)
print("saved csv:", CSV_OUT)
print("saved xlsx:", XLSX_OUT)
print("synthetic cases:", synthetic_event_df["case:concept:name"].nunique())
print("synthetic events:", len(synthetic_event_df))
print(synthetic_event_df.head())