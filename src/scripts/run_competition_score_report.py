import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.configs import RESULTS_DIR
from src.features.build_result_report import Report


def run_competition_score_report():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    score_path = os.path.join(RESULTS_DIR, "score/json")
    prompt_summary_path = os.path.join(RESULTS_DIR, "score/prompt_summary.csv")
    avg_result_method_path = os.path.join(RESULTS_DIR, "score/avg_result_method.csv")
    avg_result_prompt_path = os.path.join(RESULTS_DIR, "score/avg_result_prompt.csv")
    avg_result_method_and_prompt_path = os.path.join(
        RESULTS_DIR, "score/avg_result_method_and_prompt.csv"
    )
    all_result_path = os.path.join(RESULTS_DIR, "score/all_results.csv")

    report = Report(score_path)

    avg_result_method = report.get_avg_result_with_method()
    avg_result_prompt = report.get_avg_result_with_prompt()
    avg_result_method_and_prompt = report.get_avg_result_with_method_and_prompt()

    avg_result_method.to_csv(avg_result_method_path, index=True)
    avg_result_prompt.to_csv(avg_result_prompt_path, index=True)
    avg_result_method_and_prompt.to_csv(avg_result_method_and_prompt_path, index=True)
    report.save_results(all_result_path)

    print("Average results by method:")
    print("-" * 200)
    print(avg_result_method)
    print("\n")
    print("Average results by prompt:")
    print("-" * 200)
    print(Report.merge_sumary_prompt(prompt_summary_path, avg_result_prompt_path))
    print("\n")
    print("Average results by method and prompt:")
    print("-" * 200)
    print(
        Report.merge_sumary_prompt(
            prompt_summary_path, avg_result_method_and_prompt_path
        )
    )
    print("\n")

if __name__ == "__main__":
    run_competition_score_report()