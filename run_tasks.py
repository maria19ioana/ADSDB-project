import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_script(script_path, args=[]):
    """Executes an isolated Python script with optional arguments."""
    full_path = BASE_DIR / script_path
    
    if not full_path.exists():
        print(f"\n[ERROR] File not found: {full_path}")
        return False

    print(f"RUNNING: {script_path}")
    
    # Configure environment to include the project base directory in PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{BASE_DIR}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(BASE_DIR)
        
    try:
        command = ["python", str(full_path)] + args
        subprocess.run(command, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nScript failed with exit code {e.returncode}.")
        return False

def menu():
    while True:
        print("MULTI-MODAL TASKS MENU")
        print("TASK 1: Same-Modality Similarity Search")
        print("  1. Text Search (BGE)")
        print("  2. Image Search (CLIP)")
        print("  3. Image Search (DINOv2)")
        print("  4. Audio Search (YAMNet - requires Xeno-Canto ID)")
        print("  5. Audio Search (Google Perch - requires file path)")
        
        
        print("\nTASK 2: Multi-Modality Similarity Search")
        print("  6. Cross-Modal Search (Text/Image/Audio in CLIP space)")
        
        #print("  7. Image Captioning Testing (microsoft/git-base-coco)")
        
        print("\n0. Exit")
        
        choice = input("\nSelect an option (0-6): ").strip()
        
        if choice == "0":
            print("Exiting tasks menu.")
            break
            
        elif choice == "1":
            run_script("task1/task1_text.py")
            
        elif choice == "2":
            img_path = input("Enter absolute path for query image: ").strip()
            args = [img_path] if img_path else []
            run_script("task1/search_images.py", args)

        elif choice == "3":
                    img_path = input("Enter absolute path for query image: ").strip()
                    args = [img_path] if img_path else []
                    run_script("task1/search_images_dinov2.py", args)
            
        elif choice == "4":
            xc_id = input("Enter Xeno-Canto ID (e.g., 975331): ").strip()
            if xc_id:
                run_script("task1/search_audio_yamnet.py", [xc_id])
            else:
                print("Invalid ID. Cancelled.")

        elif choice == "5":
            audio_path = input("Enter absolute path for query audio (.wav): ").strip().strip('"')
            if audio_path:
                run_script("task1/search_audio_perch.py", [audio_path])
            else:
                print("Path cannot be empty.")
            
        elif choice == "6":
            q_type = input("Enter query type (text, image, audio): ").strip().lower()
            if q_type not in ["text", "image", "audio"]:
                print("Invalid type. Please type 'text', 'image', or 'audio'.")
                continue
            
            q_val = input("Enter text description or absolute path to the file: ").strip()
            if q_val:
                run_script("task2/search_all_v2.py", ["--type", q_type, "--query", q_val])
            else:
                print("Query cannot be empty.")
            
        #elif choice == "7":
        #    seed = input("Enter a seed for random image selection (or press Enter for random): ").strip()
        #    args = ["--seed", seed] if seed else []
        #    run_script("task3/test_microsoft_git_base_coco.py", args)
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    menu()