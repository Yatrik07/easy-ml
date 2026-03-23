import sys
import time
import threading
import subprocess

def spin(message, stop_event):
    spinner = ['-', '\\', '|', '/']
    i = 0
    while not stop_event.is_set():
        # Clear the line and print the spinner
        sys.stdout.write(f"\r{message} {spinner[i % len(spinner)]}   ")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    # Clear the spinner and show Done
    sys.stdout.write(f"\r{message} Done!       \n")
    sys.stdout.flush()

def main():
    print("====================================")
    print("    🚀 Welcome to Easy-ML Setup!   ")
    print("====================================\n")
    
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spin, args=("📦 Installing Easy-ML and full dependencies (this may take a few minutes)...", stop_event))
    spinner_thread.start()

    try:
        # Run pip install .[all]
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", ".[all]", "--upgrade"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Read silently to prevent buffer fill
        for line in process.stdout:
            pass 
            
        process.wait()
        stop_event.set()
        spinner_thread.join()
        
        if process.returncode == 0:
            print("\n✅ Installation completed successfully!")
            print("\nTo start the Easy-ML Studio, run:")
            print("    python easyml/app.py\n")
        else:
            print("\n❌ Installation failed.")
            print("Try running 'pip install .[all]' manually to see the full error logs.\n")
            
    except KeyboardInterrupt:
        stop_event.set()
        spinner_thread.join()
        print("\n\n❌ Setup cancelled by user.")
    except Exception as e:
        stop_event.set()
        spinner_thread.join()
        print(f"\n\n❌ Setup encountered an internal error: {str(e)}")

if __name__ == "__main__":
    main()
