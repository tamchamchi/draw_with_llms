import pandas as pd
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.configs import RESULTS_DIR


class Report:
     def __init__(self, results_path: str = None):
          self.results_path = results_path
          self.results = None if results_path is None else self.get_all_results()

     def get_all_results(self) -> pd.DataFrame:
          results = []
          for filename in os.listdir(self.results_path):
               if filename.endswith(".json"):
                    with open(os.path.join(self.results_path, filename), 'r') as f:
                         data = json.load(f)
                         if isinstance(data, list):  # Nếu là list of dict
                              results.extend(data)
                         elif isinstance(data, dict):  # Nếu là 1 dict đơn
                              results.append(data)
          
          return pd.DataFrame(results)
     
     def save_results(self, save_path: str):
          """
          Save the results DataFrame to a CSV file.
          
          Args:
               save_path (str): The path where the CSV file will be saved.
          """
          self.results.to_csv(save_path, index=False)
     
     def get_avg_result_with_method(self) -> pd.DataFrame:
          """
          Get the average result for a specific method.
          
          Args:
               method (str): The method name to filter results.
          
          Returns:
               pd.DataFrame: DataFrame containing the average results for the specified method.
          """
          avg_results = self.results.groupby(['method', 'model'])[['total_score', 'vqa_score', 'aesthetic_score', 'ocr_score']].mean()
          return avg_results
     def get_avg_result_with_method_and_prompt(self) -> pd.DataFrame:
          """
          Get the average result for a specific method and prompt.
          
          Args:
               method (str): The method name to filter results.
               prompt (str): The prompt name to filter results.
          
          Returns:
               pd.DataFrame: DataFrame containing the average results for the specified method and prompt.
          """
          avg_results = self.results.groupby(['method', 'id_prompt', 'model'])[['total_score', 'vqa_score', 'aesthetic_score', 'ocr_score']].mean()
          return avg_results
     
     def get_avg_result_with_prompt(self) -> pd.DataFrame:
          """
          Get the average result for a specific prompt.
          
          Args:
               prompt (str): The prompt name to filter results.
          
          Returns:
               pd.DataFrame: DataFrame containing the average results for the specified prompt.
          """
          avg_results = self.results.groupby(['id_prompt', 'model'])[['total_score', 'vqa_score', 'aesthetic_score', 'ocr_score']].mean()
          return avg_results
     
     def load_results(self, path: str) -> pd.DataFrame:
          """
          Load results from a CSV file.
          
          Args:
               path (str): The path to the CSV file.
          
          Returns:
               pd.DataFrame: DataFrame containing the loaded results.
          """
          self.results = pd.read_csv(path)
     
if __name__ == "__main__":
     score_path = os.path.join(RESULTS_DIR, "score/results")
     report = Report(score_path)
     avg_result_method = report.get_avg_result_with_method()
     avg_result_prompt = report.get_avg_result_with_prompt()
     avg_result_method_and_prompt = report.get_avg_result_with_method_and_prompt()
     print("Average results by method:")
     print("-" * 80)
     print(avg_result_method)
     print("\n")
     print("Average results by prompt:")
     print("-" * 80)
     print(avg_result_prompt)
     print("\n")
     print("Average results by method and prompt:")
     print("-" * 80)
     print(avg_result_method_and_prompt)
     print("\n")

     avg_result_method.to_csv(os.path.join(RESULTS_DIR, "score/avg_result_method.csv"), index=True)

     avg_result_prompt.to_csv(os.path.join(RESULTS_DIR, "score/avg_result_prompt.csv"), index=True)

     avg_result_method_and_prompt.to_csv(os.path.join(RESULTS_DIR, "score/avg_result_method_and_prompt.csv"), index=True)
     
     report.save_results(os.path.join(RESULTS_DIR, "score/all_results.csv"))




