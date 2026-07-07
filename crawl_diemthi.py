from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import json
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://vietnamnet.vn/giao-duc/diem-thi/tra-cuu-diem-thi-tot-nghiep-thpt/2026/{sbd}.html"
DEFAULT_OUTPUT = "scores_2026.csv"
DEFAULT_CHECKPOINT = "scores_2026.csv.checkpoint.json"
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.5
DEFAULT_CSV_FIELDS = ["sbd", "toan", "ly", "hoa", "van", "sinh", "su", "dia", "ngoai_ngu", "gdcd"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}


def parse_score_page(html, sbd):
    soup = BeautifulSoup(html, "html.parser")
    score_dict = {"sbd": sbd}

    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) != 2:
            continue

        subject = cols[0].get_text(" ", strip=True).lower()
        score = cols[1].get_text(" ", strip=True)

        if "toán" in subject:
            score_dict["toan"] = score
        elif "lí" in subject or "vật lý" in subject:
            score_dict["ly"] = score
        elif "hóa" in subject or "học" in subject:
            score_dict["hoa"] = score
        elif "ngữ văn" in subject or subject == "văn" or "văn" in subject:
            score_dict["van"] = score
        elif "sinh" in subject:
            score_dict["sinh"] = score
        elif "sử" in subject or "lịch sử" in subject:
            score_dict["su"] = score
        elif "địa" in subject or "địa lý" in subject:
            score_dict["dia"] = score
        elif "gdcd" in subject or "công dân" in subject:
            score_dict["gdcd"] = score
        elif "ngoại ngữ" in subject or "tiếng" in subject:
            score_dict["ngoai_ngu"] = score

    return score_dict if len(score_dict) > 1 else None


def fetch_score(sbd, session=None, timeout=10):
    url = BASE_URL.format(sbd=sbd)
    client = session or requests

    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            response = client.get(url, headers=HEADERS, timeout=timeout)

            if response.status_code == 200:
                return parse_score_page(response.text, sbd)

            if response.status_code not in {429, 500, 502, 503, 504}:
                return None
        except requests.RequestException:
            pass
        except Exception:
            return None

        if attempt < DEFAULT_RETRIES:
            time.sleep(DEFAULT_BACKOFF * (2 ** attempt))

    return None


def save_results(rows, output_file):
    append_results(rows, output_file)


def read_csv_fields(output_file):
    path = Path(output_file)
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_CSV_FIELDS

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header_line = f.readline().strip()
        if header_line:
            return [col.strip() for col in header_line.split(",") if col.strip()]
    except Exception:
        pass

    return DEFAULT_CSV_FIELDS


def append_results(rows, output_file):
    if not rows:
        return

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = read_csv_fields(output_file)
    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def checkpoint_file_for(output_file):
    return f"{output_file}.checkpoint.json"


def load_checkpoint(checkpoint_file):
    if not os.path.exists(checkpoint_file):
        return None
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint_file, state):
    Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def clear_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def dedupe_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        sbd = str(row.get("sbd", "")).strip()
        if not sbd or sbd in seen:
            continue
        seen.add(sbd)
        deduped.append(row)
    return deduped


def load_existing_results(output_file):
    if not os.path.exists(output_file):
        return []

    try:
        df = pd.read_csv(output_file, dtype=str)
        return df.to_dict(orient="records")
    except Exception:
        return []


def crawl_sbd_list(sbd_list, workers=8, delay=0.0, output_file=DEFAULT_OUTPUT, checkpoint_file=None, checkpoint_state=None):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for sbd in sbd_list:
            futures.append(executor.submit(fetch_score, sbd))
            if delay:
                time.sleep(delay)

        for future in as_completed(futures):
            row = future.result()
            if row:
                results.append(row)

    if output_file:
        append_results(results, output_file)
    if checkpoint_file and checkpoint_state is not None:
        save_checkpoint(checkpoint_file, checkpoint_state)
    return results


def crawl_range(start_sbd, end_sbd, workers=8, delay=0.0, output_file=DEFAULT_OUTPUT, checkpoint_file=None):
    checkpoint_file = checkpoint_file or checkpoint_file_for(output_file)
    checkpoint = load_checkpoint(checkpoint_file)
    next_sbd = start_sbd
    all_results = load_existing_results(output_file)

    if checkpoint and checkpoint.get("mode") == "range" and checkpoint.get("output") == output_file:
        next_sbd = max(start_sbd, int(checkpoint.get("next_sbd", start_sbd)))
        print(f"Resume từ SBD {str(next_sbd).zfill(8)}")

    chunk_size = 500
    while next_sbd <= end_sbd:
        chunk_end = min(next_sbd + chunk_size - 1, end_sbd)
        sbd_list = [str(i).zfill(8) for i in range(next_sbd, chunk_end + 1)]
        batch_results = crawl_sbd_list(sbd_list, workers=workers, delay=delay, output_file=None)
        all_results.extend(batch_results)
        append_results(batch_results, output_file)
        save_checkpoint(checkpoint_file, {
            "mode": "range",
            "output": output_file,
            "start_sbd": start_sbd,
            "end_sbd": end_sbd,
            "next_sbd": chunk_end + 1,
            "updated_at": time.time(),
        })
        next_sbd = chunk_end + 1

    clear_checkpoint(checkpoint_file)
    return all_results


def crawl_province(
    province_code,
    max_suffix=150000,
    workers=8,
    delay=0.0,
    empty_batches_to_stop=1,
    output_file=DEFAULT_OUTPUT,
    checkpoint_file=None,
):
    prov_prefix = f"{province_code:02d}"
    all_results = load_existing_results(output_file)
    consecutive_empty = 0
    checkpoint_file = checkpoint_file or checkpoint_file_for(output_file)
    checkpoint = load_checkpoint(checkpoint_file)
    next_batch_start = 1

    if checkpoint and checkpoint.get("mode") == "province" and checkpoint.get("province") == province_code and checkpoint.get("output") == output_file:
        next_batch_start = max(1, int(checkpoint.get("next_batch_start", 1)))
        print(f"Resume cụm {prov_prefix} từ hậu tố {next_batch_start:06d}")

    for batch_start in range(next_batch_start, max_suffix + 1, 1000):
        batch_end = min(batch_start + 999, max_suffix)
        sbd_list = [f"{prov_prefix}{i:06d}" for i in range(batch_start, batch_end + 1)]
        batch_results = crawl_sbd_list(sbd_list, workers=workers, delay=delay, output_file=None)

        if batch_results:
            all_results.extend(batch_results)
            consecutive_empty = 0
            print(f"Cụm {prov_prefix}: đã lấy {len(all_results)} bản ghi")
            append_results(batch_results, output_file)
            save_checkpoint(checkpoint_file, {
                "mode": "province",
                "output": output_file,
                "province": province_code,
                "max_suffix": max_suffix,
                "next_batch_start": batch_end + 1,
                "updated_at": time.time(),
            })
        else:
            consecutive_empty += 1
            if consecutive_empty >= empty_batches_to_stop:
                print(f"-> Cụm {prov_prefix} hết dữ liệu, chuyển cụm khác.")
                break

    clear_checkpoint(checkpoint_file)
    return all_results


def crawl_full_national(
    workers=8,
    delay=0.0,
    max_suffix=150000,
    max_province=100,
    empty_batches_to_stop=1,
    output_file=DEFAULT_OUTPUT,
    checkpoint_file=None,
):
    all_results = load_existing_results(output_file)
    checkpoint_file = checkpoint_file or checkpoint_file_for(output_file)
    checkpoint = load_checkpoint(checkpoint_file)
    start_province = 1
    start_suffix = 1

    if checkpoint and checkpoint.get("mode") == "full" and checkpoint.get("output") == output_file:
        start_province = max(1, int(checkpoint.get("next_province", 1)))
        start_suffix = max(1, int(checkpoint.get("next_suffix", 1)))
        print(f"Resume toàn quốc từ cụm {start_province:02d}, hậu tố {start_suffix:06d}")

    for province_code in range(1, max_province + 1):
        if province_code < start_province:
            continue
        prov_prefix = f"{province_code:02d}"
        print(f"\n--- Bắt đầu quét Cụm {prov_prefix} ---")
        province_all_results = []
        consecutive_empty = 0
        batch_start_iter = start_suffix if province_code == start_province else 1

        for batch_start in range(batch_start_iter, max_suffix + 1, 1000):
            batch_end = min(batch_start + 999, max_suffix)
            sbd_list = [f"{prov_prefix}{i:06d}" for i in range(batch_start, batch_end + 1)]
            batch_results = crawl_sbd_list(sbd_list, workers=workers, delay=delay, output_file=None)

            if batch_results:
                province_all_results.extend(batch_results)
                all_results.extend(batch_results)
                consecutive_empty = 0
                print(f"Cụm {prov_prefix}: đã lấy {len(province_all_results)} bản ghi")
                append_results(batch_results, output_file)
                save_checkpoint(checkpoint_file, {
                    "mode": "full",
                    "output": output_file,
                    "next_province": province_code,
                    "next_suffix": batch_end + 1,
                    "max_suffix": max_suffix,
                    "updated_at": time.time(),
                })
            else:
                consecutive_empty += 1
                if consecutive_empty >= empty_batches_to_stop:
                    print(f"-> Cụm {prov_prefix} hết dữ liệu, chuyển cụm khác.")
                    break

        start_suffix = 1

    clear_checkpoint(checkpoint_file)

    return all_results


def build_parser():
    parser = argparse.ArgumentParser(description="Cào điểm thi VietNamNet 2026.")
    parser.add_argument("--mode", choices=["one", "range", "province", "provinces", "full"], default="one")
    parser.add_argument("--sbd", help="SBD đơn lẻ, ví dụ 11001717")
    parser.add_argument("--start", type=int, help="SBD bắt đầu cho mode range")
    parser.add_argument("--end", type=int, help="SBD kết thúc cho mode range")
    parser.add_argument("--province", type=int, help="Mã tỉnh cho mode province")
    parser.add_argument(
        "--provinces",
        help="Danh sách mã tỉnh ngăn cách bằng dấu phẩy cho mode province-batch, ví dụ: 2,3,5,6",
    )
    parser.add_argument("--max-suffix", type=int, default=150000, help="Giới hạn hậu tố SBD khi quét province/full")
    parser.add_argument("--max-province", type=int, default=100, help="Số cụm tối đa khi quét toàn quốc")
    parser.add_argument(
        "--empty-batches-to-stop",
        type=int,
        default=1,
        help="Số batch rỗng liên tiếp trước khi dừng quét một cụm",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Số lần retry cho mỗi SBD")
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="Backoff cơ sở giữa các lần retry")
    return parser


def main():
    args = build_parser().parse_args()
    global DEFAULT_RETRIES, DEFAULT_BACKOFF
    DEFAULT_RETRIES = max(0, args.retries)
    DEFAULT_BACKOFF = max(0.0, args.backoff)

    if args.mode == "one":
        if not args.sbd:
            raise SystemExit("Cần truyền --sbd, ví dụ: python3 crawl_diemthi.py --mode one --sbd 11001717")
        data = crawl_sbd_list([args.sbd], workers=1, delay=0.0, output_file=args.output)
    elif args.mode == "range":
        if args.start is None or args.end is None:
            raise SystemExit("Cần truyền --start và --end")
        data = crawl_range(args.start, args.end, workers=args.workers, delay=args.delay, output_file=args.output)
    elif args.mode == "province":
        if args.province is None:
            raise SystemExit("Cần truyền --province")
        data = crawl_province(
            args.province,
            max_suffix=args.max_suffix,
            workers=args.workers,
            delay=args.delay,
            empty_batches_to_stop=args.empty_batches_to_stop,
            output_file=args.output,
        )
    elif args.mode == "full":
        data = crawl_full_national(
            workers=args.workers,
            delay=args.delay,
            max_suffix=args.max_suffix,
            max_province=args.max_province,
            empty_batches_to_stop=args.empty_batches_to_stop,
            output_file=args.output,
        )
    elif args.mode == "provinces":
        if not args.provinces:
            raise SystemExit("Cần truyền --provinces, ví dụ: --provinces 2,3,5")

        provinces = [int(item.strip()) for item in args.provinces.split(",") if item.strip()]
        data = []
        for province_code in provinces:
            print(f"\n=== Bù cụm {province_code:02d} ===")
            province_data = crawl_province(
                province_code,
                max_suffix=args.max_suffix,
                workers=args.workers,
                delay=args.delay,
                empty_batches_to_stop=args.empty_batches_to_stop,
                output_file=args.output,
            )
            data.extend(province_data)
    else:
        raise SystemExit("Mode không hợp lệ.")

    if data:
        print(f"Hoàn tất! Đã lưu {len(data)} bản ghi vào {args.output}")
    else:
        print("Không lấy được dữ liệu nào. Kiểm tra lại link và SBD.")


if __name__ == "__main__":
    main()
