import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from configs import RAW_DATA_DIR
from data.make_dataset import Data


TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
QUESTION_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/questions.parquet")

data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)

def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(data.get_questions())


if __name__ == "__main__":
    main()
