#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

# Mapping from SOR names to the CSV column suffix used by the table generator
SOR_STATUS_COLS = {
    "MeterConsolidationE4": "MeterConsolidationE4_Status",
    "FuseReplacement": "FuseReplacement_Status",
    "PlugInMeterRemoval": "PlugInMeterRemoval_Status",
}

# Mapping from Correct_Answers.csv headers to table status columns
# Correct_Answers.csv: Work_Order,Fuze Correct,Meter Correct,Plug correct
ANSWER_TO_SOR_COL = {
    "Fuze Correct": SOR_STATUS_COLS["FuseReplacement"],
    "Meter Correct": SOR_STATUS_COLS["MeterConsolidationE4"],
    "Plug correct": SOR_STATUS_COLS["PlugInMeterRemoval"],
}


def _normalize_key(k: str) -> str:
    return k.strip().lstrip("\ufeff")


def load_correct_answers(path: Path) -> Dict[str, Dict[str, str]]:
    answers: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize fieldnames to handle BOM/whitespace
        fieldname_map = {name: _normalize_key(name) for name in (reader.fieldnames or [])}
        normalized_fieldnames = list(fieldname_map.values())

        for row in reader:
            # Build a normalized row dict with cleaned keys
            clean_row = {}
            for raw_key, value in row.items():
                norm_key = fieldname_map.get(raw_key, _normalize_key(raw_key))
                clean_row[norm_key] = value

            work_order = (clean_row.get("Work_Order") or "").strip()
            if not work_order:
                continue
            answers[work_order] = {
                "Fuze Correct": (clean_row.get("Fuze Correct") or "").strip().upper(),
                "Meter Correct": (clean_row.get("Meter Correct") or "").strip().upper(),
                "Plug correct": (clean_row.get("Plug correct") or "").strip().upper(),
            }
    return answers


def load_results_table(csv_path: Path) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            work_order = (row.get("Work_Order") or "").strip()
            if not work_order:
                continue
            results[work_order] = {k: (v.strip().upper() if isinstance(v, str) else v) for k, v in row.items()}
    return results


def evaluate(results: Dict[str, Dict[str, str]], answers: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    per_sor_counts = {
        "FuseReplacement": {"correct": 0, "total": 0},
        "MeterConsolidationE4": {"correct": 0, "total": 0},
        "PlugInMeterRemoval": {"correct": 0, "total": 0},
    }

    for wo, truth in answers.items():
        res = results.get(wo)
        if not res:
            continue
        for answer_key, status_col in ANSWER_TO_SOR_COL.items():
            expected = truth.get(answer_key)
            predicted = res.get(status_col)
            if expected in ("PASS", "FAIL") and predicted in ("PASS", "FAIL"):
                sor_name = (
                    "FuseReplacement" if answer_key == "Fuze Correct" else
                    "MeterConsolidationE4" if answer_key == "Meter Correct" else
                    "PlugInMeterRemoval"
                )
                per_sor_counts[sor_name]["total"] += 1
                if expected == predicted:
                    per_sor_counts[sor_name]["correct"] += 1

    accuracies = {}
    for sor_name, counts in per_sor_counts.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracies[sor_name] = (correct / total * 100.0) if total else 0.0

    return per_sor_counts, accuracies


def display_batch_metadata(results_csv_path: Path) -> None:
    """Display batch metadata from summary.json or log file."""
    batch_dir = results_csv_path.parent
    
    # First try to load from summary.json
    summary_file = batch_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            batch_metadata = summary.get("batch_metadata", {})
            if batch_metadata:
                print("\nBatch Configuration Metadata")
                print("=" * 50)
                
                # Display client and model information
                client_type = batch_metadata.get("client_type", "Unknown")
                print(f"Client Type: {client_type.upper()}")
                
                model_config = batch_metadata.get("model_configuration", {})
                if model_config:
                    if client_type == "azure_openai":
                        print(f"Model: {model_config.get('deployment_name', 'Unknown')}")
                        print(f"Endpoint: {model_config.get('endpoint', 'Not specified')}")
                        print(f"API Version: {model_config.get('api_version', 'Unknown')}")
                    else:  # aws_bedrock
                        print(f"Model: {model_config.get('model_id', 'Unknown')}")
                        print(f"Region: {model_config.get('region', 'Unknown')}")
                
                # Display prompt configuration
                prompt_config = batch_metadata.get("prompt_configuration", {})
                if prompt_config:
                    print(f"Prompt Path: {prompt_config.get('prompt_path', 'Unknown')}")
                    if prompt_config.get('prompt_version'):
                        print(f"Prompt Version: {prompt_config.get('prompt_version')}")
                
                # Display processing configuration
                proc_config = batch_metadata.get("processing_configuration", {})
                if proc_config:
                    print(f"Batch Size: {proc_config.get('batch_size', 'Unknown')}")
                    print(f"Max Concurrent Work Orders: {proc_config.get('max_concurrent_work_orders', 'Unknown')}")
                    if proc_config.get('targeted_mode'):
                        print(f"Targeted Mode: Enabled (max {proc_config.get('max_images_per_sor', 0)} images per SOR)")
                    else:
                        print("Targeted Mode: Disabled")
                
                # Display timestamp
                metadata_time = batch_metadata.get("metadata_generated_at")
                if metadata_time:
                    print(f"Batch Generated: {metadata_time}")
                
                print("=" * 50)
                return
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Fallback: try to extract from log file
    log_file = batch_dir / "log.txt"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Extract metadata from log
            if "BATCH PROCESSING METADATA" in log_content:
                print("\nBatch Configuration Metadata (from log)")
                print("=" * 50)
                
                lines = log_content.split('\n')
                in_metadata_section = False
                
                for line in lines:
                    if "BATCH PROCESSING METADATA" in line:
                        in_metadata_section = True
                        continue
                    elif in_metadata_section and line.strip() == "=" * 80:
                        break
                    elif in_metadata_section and line.strip():
                        # Extract key information from log lines
                        if " - INFO - " in line:
                            info_part = line.split(" - INFO - ", 1)[1]
                            if any(key in info_part for key in ["Client Type:", "Prompt Configuration:", "Azure Model:", "AWS Bedrock Model:", "Enabled SOR Types:"]):
                                print(info_part)
                
                print("=" * 50)
                return
        except FileNotFoundError:
            pass
    
    # If no metadata found
    print("\nBatch Configuration Metadata")
    print("=" * 50)
    print("No batch metadata found in summary.json or log.txt")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOR results against ground truth")
    parser.add_argument(
        "--results-csv",
        type=str,
        required=False,
        help="Path to sor_results.csv (if omitted, will auto-detect the most recent in outputs)",
    )
    parser.add_argument(
        "--answers-csv",
        type=str,
        default=str(Path("tests") / "Correct_Answers.csv"),
        help="Path to Correct_Answers.csv (default: tests/Correct_Answers.csv)",
    )
    args = parser.parse_args()

    answers_path = Path(args.answers_csv)
    if not answers_path.exists():
        raise FileNotFoundError(f"Correct answers file not found: {answers_path}")

    if args.results_csv:
        results_csv_path = Path(args.results_csv)
    else:
        # Auto-detect most recent sor_results.csv under outputs/
        outputs_dir = Path("outputs")
        candidates = sorted(outputs_dir.glob("batch_results_*/sor_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No sor_results.csv found under outputs/batch_results_*/")
        results_csv_path = candidates[0]

    if not results_csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")

    answers = load_correct_answers(answers_path)
    results = load_results_table(results_csv_path)

    counts, accuracies = evaluate(results, answers)

    # Display batch metadata if available
    display_batch_metadata(results_csv_path)

    print("\nEvaluation Summary")
    print("==================")
    for sor in ("FuseReplacement", "MeterConsolidationE4", "PlugInMeterRemoval"):
        c = counts[sor]
        acc = accuracies[sor]
        print(f"{sor}: {c['correct']}/{c['total']} correct ({acc:.2f}%)")

    overall_correct = sum(counts[s]["correct"] for s in counts)
    overall_total = sum(counts[s]["total"] for s in counts)
    overall_acc = (overall_correct / overall_total * 100.0) if overall_total else 0.0
    print(f"Overall: {overall_correct}/{overall_total} correct ({overall_acc:.2f}%)\n")

    print(f"Compared results: {results_csv_path}")
    print(f"Against answers:  {answers_path}")


if __name__ == "__main__":
    main()
