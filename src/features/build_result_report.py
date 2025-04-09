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

     @staticmethod
     def merge_sumary_prompt(summary_prompt_path: str, score_path: str):
          summary_prompt_df = pd.read_csv(summary_prompt_path)

          score_df = pd.read_csv(score_path)

          merge_df = pd.merge(summary_prompt_df, score_df, how='right', on="id_prompt")

          return merge_df
     

def run():
     pd.set_option('display.max_rows', None)     
     pd.set_option('display.max_columns', None)   
     pd.set_option('display.width', None)         
     pd.set_option('display.max_colwidth', None)

     score_path = os.path.join(RESULTS_DIR, "score/json")
     prompt_summary_path = os.path.join(RESULTS_DIR, "score/prompt_summary.csv")
     avg_result_method_path = os.path.join(RESULTS_DIR, "score/avg_result_method.csv")
     avg_result_prompt_path = os.path.join(RESULTS_DIR, "score/avg_result_prompt.csv")
     avg_result_method_and_prompt_path = os.path.join(RESULTS_DIR, "score/avg_result_method_and_prompt.csv")
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
     print(Report.merge_sumary_prompt(prompt_summary_path, avg_result_method_and_prompt_path))
     print("\n")

run()





