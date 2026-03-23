import os
import pandas as pd
from ydata_profiling import ProfileReport
from easyml.config.paths import EDA_DIR
from easyml.utils.logger import write_log
import traceback

def generate_eda_report(dataframe: pd.DataFrame, dataset_name: str) -> str:
    """
    Generates an Exploratory Data Analysis (EDA) HTML report using ydata-profiling.

    Args:
        dataframe (pd.DataFrame): The pandas DataFrame to analyze.
        dataset_name (str): The name of the dataset to be used for the report filename.

    Returns:
        str: The absolute path to the generated HTML report file.
    """
    write_log(f"Starting EDA report generation for dataset {dataset_name}", "info", os.path.basename(__file__))
    try:
        eda_dir = EDA_DIR
        
        # Sanitize filename
        safe_name = dataset_name.replace(".csv", "").replace(" ", "_")
        report_path = os.path.join(eda_dir, f"{safe_name}_eda_report.html")
        
        # Generate Report
        profile = ProfileReport(dataframe, title=f"EDA Report for {dataset_name}", explorative=True, minimal=False)
        profile.to_file(report_path)
        
        write_log(f"EDA report successfully saved to {report_path}", "info", os.path.basename(__file__))
        return report_path
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        write_log(f"Failed to generate EDA report: {str(e)} - {error_traceback}", "error", os.path.basename(__file__))
        raise e
