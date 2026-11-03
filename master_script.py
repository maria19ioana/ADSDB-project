import os
import subprocess
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PIPELINE_SCRIPTS = [
    # 1. LANDING ZONE
    "landing_zone/create_bucket.py",
    "landing_zone/ingest_huggingface.py",
    "landing_zone/ingest_xenocanto.py",
    "landing_zone/persistent_landing.py",
    
    # 2. FORMATTED ZONE
    "formatted_zone/format_images.py",
    "formatted_zone/format_audio.py",
    "formatted_zone/format_text.py",
    
    # 3. TRUSTED ZONE
    "trusted_zone/trust_images.py",
    "trusted_zone/trust_audio.py",
    "trusted_zone/trust_text.py",
    
    # 4. EXPLOITATION ZONE
    "exploitation_zone/embed_images.py",
    "exploitation_zone/embed_images_dinov2.py",
    "exploitation_zone/embed_text_bge.py",
    "exploitation_zone/embed_text_clip.py",
    "exploitation_zone/embed_audio_perch.py",
    "exploitation_zone/exploit_audio_yamnet.py",

    # Task 2 collections
    "task2/create_multimodal_clip.py",
    "task2/map_all_perch.py"
]

def run_script(script_path):
    """Execute the pipeline and measure the execution time"""
    full_path = os.path.join(BASE_DIR, script_path)
    
    norm_path = os.path.normpath(full_path)
    
    if not os.path.exists(norm_path):
        print(f"\n The file was not found: {norm_path}")
        return False

    print(f" running {script_path}")
    
    start_time = time.time()
    
    try:
        env = os.environ.copy()
        
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{BASE_DIR}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = BASE_DIR
            
        subprocess.run(["python", norm_path], check=True, env=env)
        
        elapsed_time = time.time() - start_time
        print(f"\n{script_path} ran in {elapsed_time:.2f} seconds.")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n{script_path} failed (code {e.returncode}).")
        return False

def main():
    print("Starting sequential execution of Data Management Pipeline (End-to-End)\n")
    total_start = time.time()
    
    for script in PIPELINE_SCRIPTS:
        success = run_script(script)
        if not success:
            print("\nExecution of the pipeline was stopped due to an error. Fix the mentioned file and try again.")
            return

    total_elapsed = time.time() - total_start
    print(f"\nThe entire Data Management Pipeline executed successfully.")
    print(f"TTotal time: {total_elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
