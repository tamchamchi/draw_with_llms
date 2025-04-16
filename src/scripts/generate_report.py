#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse


from tabulate import tabulate  # <<<--- Thêm import tabulate
from typing import Optional  # Thêm Optional

# --- Thiết lập sys.path ---
try:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"--- Added project root to sys.path: {project_root} ---")

    # --- Import các lớp/hàm thực tế ---
    from src.reporting.report_generator import Report  # Đảm bảo đường dẫn này đúng
    from configs.configs import RESULTS_DIR  # Đảm bảo đường dẫn này đúng

except ImportError as e:
    print(f"Lỗi import: {e}")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Lỗi không tìm thấy file/thư mục: {e}")
    sys.exit(1)


def run_combined_report(json_results_dir: str,
                        output_per_file_csv: Optional[str] = None,
                        output_avg_method_csv: Optional[str] = None):
    """
    Chạy báo cáo kết hợp: tính trung bình từng file và trung bình theo method.

    Args:
        json_results_dir (str): Đường dẫn thư mục chứa file JSON kết quả.
        output_per_file_csv (Optional[str]): Đường dẫn lưu bảng trung bình từng file.
        output_avg_method_csv (Optional[str]): Đường dẫn lưu bảng trung bình theo method.
    """
    print(f"--- Khởi tạo Report và xử lý dữ liệu từ: {json_results_dir} ---")
    try:
        # Khởi tạo Report, hàm __init__ hoặc get_all_results sẽ tải dữ liệu
        report = Report(results_path=json_results_dir)

        # Gọi get_all_results một cách tường minh để chắc chắn self.results được load
        # (Nếu __init__ của bạn đã làm việc này thì bước này có thể không cần thiết,
        # nhưng nó đảm bảo dữ liệu được load để tính avg_by_method)
        if report.results is None:
            print("Đang tải lại tất cả kết quả vào DataFrame...")
            report.get_all_results()  # Gọi để load vào report.results

        if report.results is None or report.results.empty:
            print(
                "Warning: Không có dữ liệu kết quả tổng hợp để tính trung bình theo method.")
            # Vẫn có thể thử tính trung bình từng file

    except FileNotFoundError:
        print(f"Lỗi: Thư mục '{json_results_dir}' không tồn tại.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi khởi tạo Report hoặc tải dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 1. Tính và hiển thị trung bình của TỪNG FILE JSON ---
    print("\n" + "="*70)
    print("--- Điểm trung bình của từng file JSON ---")
    per_file_avg_df = None
    try:
        per_file_avg_df = report.get_average_scores_per_json_file()

        if per_file_avg_df is not None and not per_file_avg_df.empty:
            print(tabulate(per_file_avg_df, headers='keys',
                  tablefmt='psql', showindex=False, floatfmt=".4f"))
            # Lưu nếu được yêu cầu
            if output_per_file_csv:
                print(
                    f"\nĐang lưu bảng trung bình từng file vào: {output_per_file_csv}")
                output_dir = os.path.dirname(output_per_file_csv)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                try:
                    per_file_avg_df.to_csv(
                        output_per_file_csv, index=False, encoding='utf-8')
                    print("Lưu thành công.")
                except Exception as e_save:
                    print(f"Lỗi khi lưu CSV từng file: {e_save}")
        else:
            print("Không có dữ liệu trung bình của từng file để hiển thị.")
    except AttributeError:
        print("Lỗi: Thiếu phương thức 'get_average_scores_per_json_file' trong lớp Report.")
    except Exception as e:
        print(f"Lỗi khi tính/hiển thị trung bình từng file: {e}")
        import traceback
        traceback.print_exc()

    # --- 2. Tính và hiển thị trung bình theo METHOD (tổng hợp) ---
    print("\n" + "="*70)
    print("--- Điểm trung bình tổng hợp theo Method và Model ---")
    avg_method_df = None
    try:
        # Kiểm tra lại report.results trước khi gọi get_avg_result_with_method
        if report.results is not None and not report.results.empty:
            avg_method_df = report.get_avg_result_with_method()
        else:
            print("Không có dữ liệu tổng hợp để tính trung bình theo method.")

        if avg_method_df is not None and not avg_method_df.empty:
            print(tabulate(avg_method_df, headers='keys', tablefmt='psql',
                  showindex=True, floatfmt=".4f"))  # showindex=True
            # Lưu nếu được yêu cầu
            if output_avg_method_csv:
                print(
                    f"\nĐang lưu bảng trung bình theo method vào: {output_avg_method_csv}")
                output_dir = os.path.dirname(output_avg_method_csv)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                try:
                    avg_method_df.to_csv(
                        output_avg_method_csv, index=True, encoding='utf-8')
                    print("Lưu thành công.")  # index=True
                except Exception as e_save:
                    print(f"Lỗi khi lưu CSV trung bình method: {e_save}")
        else:
            # Đã in cảnh báo ở trên nếu report.results rỗng
            if report.results is not None and not report.results.empty:
                print(
                    "Không tính được trung bình theo method (có thể do thiếu cột hoặc lỗi khác).")

    except AttributeError:
        print("Lỗi: Thiếu phương thức 'get_avg_result_with_method' trong lớp Report.")
    except Exception as e:
        print(f"Lỗi khi tính/hiển thị trung bình theo method: {e}")
        import traceback
        traceback.print_exc()

    # --- Không còn phần liệt kê chi tiết từng item trong JSON ---

    print("\n--- Script hoàn thành ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chạy báo cáo: Trung bình từng file và Trung bình theo method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--json_dir',
        type=str,
        default=os.path.join(RESULTS_DIR, "score/json"),  # Đường dẫn mặc định
        help='Đường dẫn đến thư mục chứa các file kết quả JSON.'
    )
    parser.add_argument(
        '--output_per_file_csv',  # Đổi tên để rõ ràng hơn
        type=str,
        default=None,
        help='(Tùy chọn) Đường dẫn file CSV để lưu bảng kết quả trung bình từng file.'
    )
    parser.add_argument(
        '--output_avg_method_csv',  # Thêm tham số này
        type=str,
        default=None,
        help='(Tùy chọn) Đường dẫn file CSV để lưu bảng kết quả trung bình theo method.'
    )

    args = parser.parse_args()

    run_combined_report(
        args.json_dir, args.output_per_file_csv, args.output_avg_method_csv)
