#!/usr/bin/env python3
"""
Script to run both gestational age and birth weight pipelines simultaneously.
"""

import subprocess
import time
import os

def run_pipeline(target_type):
    """Run a single pipeline with the specified target type."""
    print(f"Starting {target_type} pipeline...")
    cmd = ["python3", "main.py", target_type]
    
    # Create log file
    log_file = f"{target_type}_pipeline.log"
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"  Process ID: {process.pid}")
        print(f"  Log file: {log_file}")
        return process

def main():
    print("=== Running Both Pipelines Simultaneously ===\n")
    
    # Start both pipelines
    ga_process = run_pipeline("gestational_age")
    bw_process = run_pipeline("birth_weight")
    
    print(f"\nBoth pipelines started!")
    print(f"Gestational Age PID: {ga_process.pid}")
    print(f"Birth Weight PID: {bw_process.pid}")
    print("\nMonitoring progress...")
    
    # Monitor processes
    while True:
        ga_status = ga_process.poll()
        bw_status = bw_process.poll()
        
        if ga_status is not None and bw_status is not None:
            print("\nBoth pipelines completed!")
            print(f"Gestational Age exit code: {ga_status}")
            print(f"Birth Weight exit code: {bw_status}")
            break
        elif ga_status is not None:
            print(f"Gestational Age pipeline completed (exit code: {ga_status})")
            break
        elif bw_status is not None:
            print(f"Birth Weight pipeline completed (exit code: {bw_status})")
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    print("\nCheck the log files for detailed output:")
    print("- gestational_age_pipeline.log")
    print("- birth_weight_pipeline.log")

if __name__ == "__main__":
    main() 