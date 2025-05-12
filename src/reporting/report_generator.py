# src/reporting/report.py

import json
import os
import pandas as pd
from tabulate import tabulate  # <<<--- Đảm bảo đã import tabulate
from typing import Dict, List, Union, Any, Optional


class Report:
    # ... (__init__, get_all_results, get_json_data_by_file giữ nguyên) ...
    def __init__(self, results_path: Optional[str] = None):
        self.results_path = results_path
        self.results: Optional[pd.DataFrame] = None
        if results_path is not None:
            try:
                self.results = self.get_all_results()
            except FileNotFoundError:
                print(f"Warning: results_path '{results_path}' not found.")
                self.results = None
            except Exception as e:
                print(f"Warning: Error loading results: {e}")
                self.results = None

    def get_all_results(self) -> Optional[pd.DataFrame]:
        if not self.results_path or not os.path.isdir(self.results_path):
            return None
        results_list = []
        try:
            for filename in os.listdir(self.results_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.results_path, filename)
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            results_list.extend(data)
                        elif isinstance(data, dict):
                            results_list.append(data)
                    except Exception as e:
                        print(
                            f"  Warning: Skipping {filename} due to error: {e}")
        except Exception as e:
            print(f"Error reading directory {self.results_path}: {e}")
            return None
        if not results_list:
            return pd.DataFrame()
        return pd.DataFrame(results_list)

    # --- PHƯƠNG THỨC MỚI ĐỂ TÍNH TRUNG BÌNH TỪNG FILE ---
    def get_average_scores_per_json_file(self) -> Optional[pd.DataFrame]:
        """
        Tính điểm trung bình cho nội dung bên trong từng file JSON riêng lẻ
        trong thư mục results_path.

        Returns:
            Optional[pd.DataFrame]: DataFrame với mỗi hàng là một file JSON,
                                    bao gồm tên file và điểm trung bình các cột score.
                                    Trả về None nếu có lỗi thư mục.
        """
        if not self.results_path or not os.path.isdir(self.results_path):
            print(f"Lỗi: results_path '{self.results_path}' không phải là thư mục hợp lệ.")
            return None

        per_file_avg_list = []
        # Các cột điểm cần tính trung bình
        # score_columns = ["total_score", "vqa_score", "aesthetic_score", "ocr_score"]
        # Có thể thêm các cột khác nếu muốn, ví dụ clip_similarity
        score_columns = ["total_score", "vqa_score", "vqa_origin", "aesthetic_score", "aesthetic_origin", "ocr_score", "text_alignment_score", "size"]

        print(f"--- Đang tính điểm trung bình cho từng file JSON trong: {self.results_path} ---")

        for filename in sorted(os.listdir(self.results_path)): # Sắp xếp tên file cho dễ nhìn
            if filename.endswith(".json"):
                file_path = os.path.join(self.results_path, filename)
                # Khởi tạo dict kết quả cho file này, điểm mặc định là NaN (Not a Number)
                file_avg_data = {"filename": filename}
                for col in score_columns:
                    file_avg_data[f"avg_{col}"] = pd.NA # Dùng pd.NA tốt hơn None cho pandas

                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = json.load(f)

                    # Xử lý dữ liệu đã load
                    if isinstance(data, dict):
                        # Trường hợp file JSON chứa 1 dict kết quả duy nhất
                        print(f"  Processing {filename} (single dict)...")
                        for col in score_columns:
                            # Lấy giá trị, nếu thiếu key hoặc không phải số sẽ là NA
                            file_avg_data[f"avg_{col}"] = pd.to_numeric(data.get(col), errors='coerce')

                    elif isinstance(data, list) and data:
                        # Trường hợp file JSON chứa list các dict kết quả
                        print(f"  Processing {filename} (list of {len(data)} items)...")
                        if isinstance(data[0], dict):
                            temp_df = pd.DataFrame(data)
                            existing_score_cols = [col for col in score_columns if col in temp_df.columns]
                            if existing_score_cols:
                                numeric_df = temp_df[existing_score_cols].apply(pd.to_numeric, errors='coerce')
                                file_means = numeric_df.mean(skipna=True).to_dict() # Tính trung bình, bỏ qua NaN
                                for col, mean_val in file_means.items():
                                    file_avg_data[f"avg_{col}"] = mean_val
                            else:
                                print(f"    Warning: Không tìm thấy cột điểm nào trong list của file {filename}.")
                        else:
                             print(f"    Warning: List trong file {filename} không chứa dictionary.")
                    else: # Trường hợp list rỗng hoặc kiểu dữ liệu khác
                         print(f"    Warning: Kiểu dữ liệu không mong đợi ({type(data).__name__}) hoặc list rỗng trong file {filename}.")

                except json.JSONDecodeError:
                    print(f"  Error: Bỏ qua file {filename} do lỗi giải mã JSON.")
                except Exception as e:
                    print(f"  Error: Bỏ qua file {filename} do lỗi không xác định: {e}")

                # Thêm kết quả (hoặc lỗi/NaN) của file này vào list tổng hợp
                per_file_avg_list.append(file_avg_data)

        if not per_file_avg_list:
            print("Không xử lý được file JSON nào.")
            return pd.DataFrame() # Trả về DataFrame rỗng

        # Tạo DataFrame cuối cùng
        final_df = pd.DataFrame(per_file_avg_list).sort_values("avg_total_score", ascending=False)
        return final_df

    def get_json_data_by_file(self) -> Dict[str, Union[List, Dict, Any]]:
        if not self.results_path:
            raise ValueError("results_path was not provided.")
        if not os.path.isdir(self.results_path):
            raise FileNotFoundError(f"Path not found: {self.results_path}")
        data_by_file = {}
        for filename in os.listdir(self.results_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.results_path, filename)
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = json.load(f)
                    data_by_file[filename] = data
                except Exception as e:
                    data_by_file[filename] = {"error": str(e)}
        return data_by_file

    # --- PHƯƠNG THỨC display_json_data_by_file ĐƯỢC CẬP NHẬT ---

    def display_json_data_by_file(self, max_items_per_file: int = 5):
        """
        In nội dung của từng file JSON, hiển thị dạng bảng nếu nội dung
        là list các dictionary hoặc một dictionary đơn.
        """
        try:
            data_map = self.get_json_data_by_file()
            if not data_map:
                print("Không tìm thấy hoặc không tải được file JSON nào.")
                return

            print("\n--- Contents of Individual JSON Files (Tabulated) ---")
            for filename, data in data_map.items():
                print(f"\n=== File: {filename} ===")
                if isinstance(data, dict) and "error" in data:
                    print(f"  ERROR loading this file: {data['error']}")
                    continue

                # --- Xử lý nếu data là LIST ---
                if isinstance(data, list):
                    print(f"  Type: List ({len(data)} items)")
                    if len(data) > 0:
                        # Kiểm tra item đầu tiên có phải dict không để biết có thể tạo bảng không
                        if isinstance(data[0], dict):
                            try:
                                # Tạo DataFrame từ các item đầu tiên
                                display_data = data[:max_items_per_file]
                                df_display = pd.DataFrame(display_data)
                                print(tabulate(
                                    df_display, headers='keys', tablefmt='psql', showindex=False, floatfmt=".4f"))
                                if len(data) > max_items_per_file:
                                    print(
                                        f"  ... (and {len(data) - max_items_per_file} more items)")
                            except Exception as e_df:
                                # Nếu lỗi tạo DataFrame (ví dụ các dict không đồng nhất key) thì in kiểu cũ
                                print(
                                    f"  (Could not create table: {e_df}. Displaying raw items instead.)")
                                for i, item in enumerate(data[:max_items_per_file]):
                                    print(
                                        f"  Item {i}: {json.dumps(item, indent=2, ensure_ascii=False)}")
                                if len(data) > max_items_per_file:
                                    print(
                                        f"  ... (and {len(data) - max_items_per_file} more items)")
                        else:
                            # Nếu list không chứa dict, in kiểu cũ
                            print(
                                "  (List does not contain dictionaries, displaying raw items)")
                            for i, item in enumerate(data[:max_items_per_file]):
                                print(f"  Item {i}: {item}")
                            if len(data) > max_items_per_file:
                                print(
                                    f"  ... (and {len(data) - max_items_per_file} more items)")
                    else:
                        print("  (List is empty)")

                # --- Xử lý nếu data là DICTIONARY ---
                elif isinstance(data, dict):
                    print(f"  Type: Dictionary ({len(data)} keys)")
                    if data:  # Nếu dict không rỗng
                        # Tạo list of lists để dùng tabulate
                        table_data = [[key, str(value)[:100] + ('...' if len(str(value)) > 100 else '')]  # Cắt bớt value dài
                                      for key, value in list(data.items())[:max_items_per_file]]
                        print(tabulate(table_data, headers=[
                              'Key', 'Value (Preview)'], tablefmt='psql'))
                        if len(data) > max_items_per_file:
                            print(
                                f"  ... (and {len(data) - max_items_per_file} more keys)")
                    else:
                        print("  (Dictionary is empty)")

                # --- Xử lý các kiểu dữ liệu khác ---
                else:
                    print(f"  Type: {type(data).__name__}")
                    # In một phần nội dung
                    print(f"  Content: {str(data)[:200]}...")

            print("\n---------------------------------------------")

        except (FileNotFoundError, ValueError) as e:
            print(f"Error displaying JSON data: {e}")
        except Exception as e:
            print(f"Unexpected error during display: {e}")
            import traceback
            traceback.print_exc()

    # ... (Các phương thức khác: save_results, get_avg_*, load_results, merge_sumary_prompt giữ nguyên) ...
    def save_results(self, save_path: str):
        if self.results is None or self.results.empty:
            print("Warning: No results data to save.")
            return
        print(f"Saving results to CSV: {save_path}")
        try:
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            self.results.to_csv(save_path, index=False, encoding='utf-8')
            print("Save successful.")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

    def get_avg_result_with_method(self) -> Optional[pd.DataFrame]:
        if self.results is None or self.results.empty:
            return None
        required_cols = ["method", "model", "total_score",
                         "vqa_score", "aesthetic_score", "ocr_score"]
        if not all(col in self.results.columns for col in required_cols):
            return None
        try:
            numeric_cols = ["total_score", "vqa_score",
                            "aesthetic_score", "ocr_score"]
            for col in numeric_cols:
                self.results[col] = pd.to_numeric(
                    self.results[col], errors='coerce')
            avg_results = self.results.groupby(["method", "model"])[
                numeric_cols].mean()
            return avg_results
        except Exception as e:
            print(f"Error in get_avg_result_with_method: {e}")
            return None

    def get_avg_result_with_method_and_prompt(self) -> Optional[pd.DataFrame]:
        if self.results is None or self.results.empty:
            return None
        required_cols = ["method", "id_prompt", "model",
                         "total_score", "vqa_score", "aesthetic_score", "ocr_score"]
        if not all(col in self.results.columns for col in required_cols):
            return None
        try:
            numeric_cols = ["total_score", "vqa_score",
                            "aesthetic_score", "ocr_score"]
            for col in numeric_cols:
                self.results[col] = pd.to_numeric(
                    self.results[col], errors='coerce')
            avg_results = self.results.groupby(["method", "id_prompt", "model"])[
                numeric_cols].mean()
            return avg_results
        except Exception as e:
            print(f"Error in get_avg_result_with_method_and_prompt: {e}")
            return None

    def get_avg_result_with_prompt(self) -> Optional[pd.DataFrame]:
        if self.results is None or self.results.empty:
            return None
        required_cols = ["id_prompt", "model", "total_score",
                         "vqa_score", "aesthetic_score", "ocr_score"]
        if not all(col in self.results.columns for col in required_cols):
            return None
        try:
            numeric_cols = ["total_score", "vqa_score",
                            "aesthetic_score", "ocr_score"]
            for col in numeric_cols:
                self.results[col] = pd.to_numeric(
                    self.results[col], errors='coerce')
            avg_results = self.results.groupby(["id_prompt", "model"])[
                numeric_cols].mean()
            return avg_results
        except Exception as e:
            print(f"Error in get_avg_result_with_prompt: {e}")
            return None

    def load_results(self, path: str):
        print(f"Loading results from CSV: {path}")
        if not os.path.exists(path):
            print(f"Error: CSV file not found: {path}")
            self.results = None
            return
        try:
            self.results = pd.read_csv(path)
            print(f"Loaded {len(self.results)} rows.")
        except Exception as e:
            print(f"Error loading CSV file {path}: {e}")
            self.results = None

    @staticmethod
    def merge_sumary_prompt(summary_prompt_path: str, score_path: str) -> Optional[pd.DataFrame]:
        print(
            f"Merging summary: {summary_prompt_path} and scores: {score_path}")
        if not os.path.exists(summary_prompt_path):
            print(f"Error: Summary file not found: {summary_prompt_path}")
            return None
        if not os.path.exists(score_path):
            print(f"Error: Score file not found: {score_path}")
            return None
        try:
            summary_prompt_df = pd.read_csv(summary_prompt_path)
            score_df = pd.read_csv(score_path)
            merge_col = "id_prompt"  # Hoặc 'id', 'id_desc'
            if merge_col not in summary_prompt_df.columns:
                print(f"Error: Column '{merge_col}' not in summary.")
                return None
            if merge_col not in score_df.columns:
                print(f"Error: Column '{merge_col}' not in score.")
                return None
            merge_df = pd.merge(summary_prompt_df, score_df,
                                how="right", on=merge_col)
            return merge_df
        except Exception as e:
            print(f"Error during merge: {e}")
            return None
