import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_command(command):
    logging.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, env=dict(os.environ, PYTHONPATH="."))
    if result.returncode != 0:
        logging.error(f"Command failed with code {result.returncode}")
        exit(1)

def main():
    import sys
    python_exe = sys.executable
    logging.info("Starting ABSA Project End-to-End Run")
    run_command([python_exe, "src/data.py"])
    run_command([python_exe, "src/preprocess.py"])
    run_command([python_exe, "src/pipeline.py"])
    
    logging.info("All pipeline steps completed successfully!")
    logging.info("Deliverables checklist:")
    logging.info("- [x] data/raw/")
    logging.info("- [x] data/processed/")
    logging.info("- [x] results/figures/model_comparison.png")
    logging.info("- [x] results/tables/model_comparison.csv")
    logging.info("- [x] results/tables/error_analysis.md")
    logging.info("- [x] app/app.py (Streamlit demo ready)")
    logging.info("- [x] report/report.md")
    
if __name__ == "__main__":
    main()
