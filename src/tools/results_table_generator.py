from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import json

from ..config import ConfigManager


class ResultsTableGenerator:
    """Converts SOR analysis JSON results into structured table format."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.table_config = self._get_table_config()
    
    def _get_table_config(self) -> Dict[str, Any]:
        """Get table generation configuration from SOR analysis config."""
        sor_config = self.config_manager.get_config("sor_analysis_config")
        table_config = sor_config.get("sor_analysis", {}).get("table_generation", {})
        
        # Ensure we have the configuration, fallback to defaults only if missing
        if not table_config:
            self.logger.warning("No table_generation config found, using defaults")
            table_config = {
                "boolean_fields": {
                    "AsbestosBagAndBoard": "Valid_Claim",
                    "CertificateOfCompliance": "Valid_Certificate",
                    "FuseReplacement": "valid_claim",
                    "MeterConsolidationE4": "consolidation",
                    "PlugInMeterRemoval": "meters_removed",
                    "ServiceProtectionDevices": "devices_added",
                    "SwitchInstallation": "switch_installed",
                    "NeutralLinkInstallation": "neutral_link_installed",
                    "meter_reading": "most_likely"
                },
                "notes_fields": {
                    "AsbestosBagAndBoard": "Notes",
                    "CertificateOfCompliance": "Notes",
                    "FuseReplacement": "notes",
                    "MeterConsolidationE4": "notes",
                    "PlugInMeterRemoval": "notes",
                    "ServiceProtectionDevices": "notes",
                    "SwitchInstallation": "notes",
                    "NeutralLinkInstallation": "notes",
                    "meter_reading": "notes"
                },
                "additional_fields": {
                    "AsbestosBagAndBoard": ["National_Meter_Identifier", "Date_On_Bag", "Asbestos_bag_image", "Meter_board_image"],
                    "CertificateOfCompliance": ["Certificate_type", "Electrical_Work_Present", "Certificate_image"],
                    "FuseReplacement": ["fuse_count", "is_even_count", "fuse_image"],
                    "MeterConsolidationE4": ["init_count", "final_count", "init_image", "final_image"],
                    "PlugInMeterRemoval": ["init_count", "final_count", "init_image", "final_image"],
                    "ServiceProtectionDevices": ["init_count", "final_count", "init_image", "final_image"],
                    "SwitchInstallation": ["switch_count", "init_image", "final_image"],
                    "NeutralLinkInstallation": ["init_count", "final_count", "init_image", "final_image"],
                    "meter_reading": ["possible_values", "meter_type", "confidence"]
                },
                "output_formats": ["csv", "excel"],
                "include_metadata": True,
                "include_costs": True,
                "sort_by_work_order": True
            }
        
        return table_config
    
    def generate_table_from_batch_results(self, batch_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate a structured table from batch SOR analysis results.
        
        Args:
            batch_results: Results from batch work order processing
            
        Returns:
            Pandas DataFrame with work orders as rows and SOR results as columns
        """
        work_order_results = batch_results.get("work_order_results", {})
        
        if not work_order_results:
            self.logger.warning("No work order results found")
            return pd.DataFrame()
        
        # Get SOR types from configuration
        sor_config = self.config_manager.get_config("sor_analysis_config")
        sor_types = list(sor_config.get("sor_analysis", {}).get("sor_types", {}).keys())
        
        # Initialize table data
        table_data = []
        
        for work_order_number, work_order_result in work_order_results.items():
            row_data = {"Work_Order": work_order_number}
            
            # Check if work order processing failed
            if "error" in work_order_result:
                # Add error row
                for sor_type in sor_types:
                    row_data[f"{sor_type}_Status"] = "ERROR"
                    row_data[f"{sor_type}_Confidence"] = 0  # Add confidence as 0 for errors
                    row_data[f"{sor_type}_Notes"] = work_order_result["error"]
                table_data.append(row_data)
                continue
            
            # Extract SOR results
            sor_results = work_order_result.get("sor_results", {})
            
            for sor_type in sor_types:
                sor_result = sor_results.get(sor_type, {})
                
                # Check if SOR analysis failed
                if "error" in sor_result:
                    row_data[f"{sor_type}_Status"] = "ERROR"
                    row_data[f"{sor_type}_Confidence"] = 0  # Add confidence as 0 for errors
                    row_data[f"{sor_type}_Notes"] = sor_result["error"]
                    continue
                
                # Extract boolean status
                boolean_field = self.table_config["boolean_fields"].get(sor_type)
                if boolean_field and boolean_field in sor_result:
                    status = sor_result[boolean_field]
                    row_data[f"{sor_type}_Status"] = "PASS" if status else "FAIL"
                else:
                    # If boolean field is missing, treat as FAIL
                    row_data[f"{sor_type}_Status"] = "FAIL"
                    self.logger.warning(f"Missing boolean field '{boolean_field}' for {sor_type} in work order {work_order_number}, treating as FAIL")
                
                # Extract confidence score - SPECIAL HANDLING
                confidence_score = sor_result.get("confidence_score", 0)
                row_data[f"{sor_type}_Confidence"] = confidence_score
                
                # Add confidence level indicator (High/Medium/Low)
                if confidence_score >= 80:
                    confidence_level = "HIGH"
                elif confidence_score >= 60:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
                row_data[f"{sor_type}_Conf_Level"] = confidence_level
                
                # Extract notes
                notes_field = self.table_config["notes_fields"].get(sor_type)
                if notes_field and notes_field in sor_result:
                    row_data[f"{sor_type}_Notes"] = str(sor_result[notes_field])
                else:
                    row_data[f"{sor_type}_Notes"] = ""
                
                # Extract additional fields (but skip confidence_score as we handled it)
                additional_fields = self.table_config["additional_fields"].get(sor_type, [])
                for field in additional_fields:
                    if field == "confidence_score":
                        continue  # Already handled above
                    if field in sor_result:
                        row_data[f"{sor_type}_{field}"] = str(sor_result[field])
                    else:
                        row_data[f"{sor_type}_{field}"] = ""
                        self.logger.debug(f"Missing field '{field}' for {sor_type} in work order {work_order_number}")
                
                # Log all available fields for debugging
                if sor_result:
                    self.logger.debug(f"Available fields for {sor_type} in {work_order_number}: {list(sor_result.keys())}")
            
            # Add metadata if configured
            if self.table_config["include_metadata"]:
                summary = work_order_result.get("summary", {})
                row_data["Analysis_Timestamp"] = summary.get("analysis_timestamp", "")
                row_data["Total_Cost"] = summary.get("total_cost", 0)
                row_data["Input_Tokens"] = summary.get("input_tokens", 0)
                row_data["Output_Tokens"] = summary.get("output_tokens", 0)
            
            table_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Sort by work order number if configured
        if self.table_config["sort_by_work_order"] and not df.empty:
            df = df.sort_values("Work_Order")
        
        # Reorder columns to group by SOR type
        if not df.empty:
            ordered_columns = ["Work_Order"]
            for sor_type in sor_types:
                # Order: Status, Confidence, Conf_Level, additional fields, Notes
                if f"{sor_type}_Status" in df.columns:
                    ordered_columns.append(f"{sor_type}_Status")
                if f"{sor_type}_Confidence" in df.columns:
                    ordered_columns.append(f"{sor_type}_Confidence")
                if f"{sor_type}_Conf_Level" in df.columns:
                    ordered_columns.append(f"{sor_type}_Conf_Level")
                
                # Add additional fields
                additional_fields = self.table_config["additional_fields"].get(sor_type, [])
                for field in additional_fields:
                    if field != "confidence_score" and f"{sor_type}_{field}" in df.columns:
                        ordered_columns.append(f"{sor_type}_{field}")
                
                if f"{sor_type}_Notes" in df.columns:
                    ordered_columns.append(f"{sor_type}_Notes")
            
            # Add metadata columns at the end
            for col in df.columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            df = df[ordered_columns]
        
        # Log summary of extracted fields
        if not df.empty:
            self.logger.info(f"Generated table with {len(df)} work orders and {len(df.columns)} columns")
            
            # Calculate and log average confidence scores
            for sor_type in sor_types:
                conf_col = f"{sor_type}_Confidence"
                if conf_col in df.columns:
                    avg_conf = df[conf_col].mean()
                    self.logger.info(f"Average confidence for {sor_type}: {avg_conf:.1f}")
        else:
            self.logger.warning("Generated empty table")
        
        return df
    
    def save_table(self, df: pd.DataFrame, output_path: str, format: str = "csv") -> str:
        """
        Save the generated table to file.
        
        Args:
            df: Pandas DataFrame to save
            output_path: Base path for output file
            format: Output format ('csv' or 'excel')
            
        Returns:
            Path to saved file
        """
        if df.empty:
            self.logger.warning("No data to save")
            return ""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "csv":
            file_path = output_path.with_suffix(".csv")
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved CSV table to: {file_path}")
            
        elif format.lower() == "excel":
            file_path = output_path.with_suffix(".xlsx")
            df.to_excel(file_path, index=False, engine="openpyxl")
            self.logger.info(f"Saved Excel table to: {file_path}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(file_path)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from the table.
        
        Args:
            df: Pandas DataFrame with SOR results
            
        Returns:
            Dict with summary statistics
        """
        if df.empty:
            return {"error": "No data available"}
        
        # Get SOR types from column names
        sor_columns = [col for col in df.columns if col.endswith("_Status")]
        sor_types = [col.replace("_Status", "") for col in sor_columns]
        
        summary = {
            "total_work_orders": len(df),
            "sor_types": sor_types,
            "sor_results": {}
        }
        
        for sor_type in sor_types:
            status_col = f"{sor_type}_Status"
            conf_col = f"{sor_type}_Confidence"
            
            if status_col in df.columns:
                status_counts = df[status_col].value_counts()
                sor_summary = {
                    "pass": int(status_counts.get("PASS", 0)),
                    "fail": int(status_counts.get("FAIL", 0)),
                    "error": int(status_counts.get("ERROR", 0)),
                    "unknown": int(status_counts.get("UNKNOWN", 0))
                }
                
                # Add confidence statistics
                if conf_col in df.columns:
                    # Filter out error rows (confidence = 0)
                    valid_conf = df[df[status_col] != "ERROR"][conf_col] if len(df[df[status_col] != "ERROR"]) > 0 else df[conf_col]
                    
                    if len(valid_conf) > 0:
                        sor_summary["confidence_stats"] = {
                            "average": float(valid_conf.mean()),
                            "min": float(valid_conf.min()),
                            "max": float(valid_conf.max()),
                            "high_confidence_count": int((valid_conf >= 80).sum()),
                            "medium_confidence_count": int(((valid_conf >= 60) & (valid_conf < 80)).sum()),
                            "low_confidence_count": int((valid_conf < 60).sum())
                        }
                
                summary["sor_results"][sor_type] = sor_summary
        
        # Add metadata if available
        if "Total_Cost" in df.columns:
            summary["total_cost"] = float(df["Total_Cost"].sum())
        if "Input_Tokens" in df.columns:
            summary["total_input_tokens"] = int(df["Input_Tokens"].sum())
        if "Output_Tokens" in df.columns:
            summary["total_output_tokens"] = int(df["Output_Tokens"].sum())
        
        return summary
    
    def save_summary_report(self, summary: Dict[str, Any], output_path: str) -> str:
        """
        Save summary statistics to a JSON file.
        
        Args:
            summary: Summary statistics dict
            output_path: Base path for output file
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path.with_suffix(".json")
        
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved summary report to: {file_path}")
        return str(file_path) 