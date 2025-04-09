from pathlib import Path
import os

PROJ_DIR = Path(__file__).parents[1]

MODEL_DIR = os.path.join(PROJ_DIR, 'models')

DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'internal')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')



# print(f"Project Dir: {PROJ_DIR}")
# print(f"Model Dir: {MODEL_DIR}")
# print(f"Data Dir: {DATA_DIR}")
# print(f"Processed Data Dir: {PROCESSED_DATA_DIR}")
# print(f"Raw Data Dir: {RAW_DATA_DIR}")
# print(f"Internal Data Dir: {INTERNAL_DATA_DIR}")
