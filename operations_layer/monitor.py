import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_DIR / "operations_layer" / "monitoring_clean.csv"


def get_processes(process):
    """Get the pipeline process and its child processes."""
    try:
        return [process] + process.children(recursive=True)
    except psutil.Error:
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python operations_layer/monitor.py <pipeline_script.py>")
        return

    pipeline_script = PROJECT_DIR / sys.argv[1]

    if not pipeline_script.is_file():
        print(f"Pipeline script not found: {pipeline_script}")
        return

    print("Starting pipeline monitoring...")

    start_time = time.time()

    # Start the orchestrator
    pipeline = subprocess.Popen(
        [sys.executable, str(pipeline_script)],
        cwd=PROJECT_DIR
    )

    process = psutil.Process(pipeline.pid)
    monitored = set()

    max_cpu = 0
    max_ram = 0
    samples = []

    try:
        while pipeline.poll() is None:
            cpu = 0
            ram = 0

            for child in get_processes(process):
                try:
                    if child.pid not in monitored:
                        child.cpu_percent(interval=None)
                        monitored.add(child.pid)

                    cpu += child.cpu_percent(interval=None)
                    ram += child.memory_info().rss

                except psutil.Error:
                    continue

            ram_mb = ram / (1024 * 1024)
            disk = shutil.disk_usage(PROJECT_DIR)
            disk_percent = disk.used / disk.total * 100

            elapsed = time.time() - start_time

            max_cpu = max(max_cpu, cpu)
            max_ram = max(max_ram, ram_mb)

            samples.append({
                "time_seconds": round(elapsed, 2),
                "cpu_percent": round(cpu, 2),
                "ram_mb": round(ram_mb, 2),
                "disk_percent": round(disk_percent, 2)
            })

            print(
                f"Time: {elapsed:.0f}s | "
                f"CPU: {cpu:.1f}% | "
                f"RAM: {ram_mb:.1f} MB | "
                f"Disk: {disk_percent:.1f}%",
                flush=True
            )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoring interrupted.")
        pipeline.terminate()

    finally:
        pipeline.wait()

        total_time = time.time() - start_time

        # Save monitoring results
        with open(OUTPUT_FILE, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "time_seconds",
                    "cpu_percent",
                    "ram_mb",
                    "disk_percent"
                ]
            )
            writer.writeheader()
            writer.writerows(samples)

        status = "SUCCESS" if pipeline.returncode == 0 else "FAILED"

        print(f"Status: {status}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Peak CPU: {max_cpu:.2f}%")
        print(f"Peak RAM: {max_ram:.2f} MB")
        print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
