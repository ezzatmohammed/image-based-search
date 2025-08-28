import os
import sys
import subprocess
import logging
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

def start_uvicorn():
    """Start the FastAPI app with uvicorn"""
    try:
        logging.info("Starting FastAPI server with uvicorn...")
        # Run in a subprocess so we can run ngrok afterwards
        uvicorn_process = subprocess.Popen(
            ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return uvicorn_process
    except Exception as e:
        logging.error(f"Failed to start uvicorn: {e}")
        sys.exit(1)

def start_ngrok_static():
    """Start ngrok with a static reserved domain"""
    try:
        logging.info("Starting ngrok with static domain lately-humane-swan.ngrok-free.app...")
        ngrok_process = subprocess.Popen(
            ["D:\\NGROK\\ngrok.exe", "http", "--domain=lately-humane-swan.ngrok-free.app", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)
        logging.info("ngrok tunnel is now running at https://lately-humane-swan.ngrok-free.app")
        logging.info("API docs (if applicable) at: https://lately-humane-swan.ngrok-free.app/docs")
        return ngrok_process
    except Exception as e:
        logging.error(f"Failed to start ngrok with static domain: {e}")
        sys.exit(1)

def main():
    """Main function to run the application with ngrok"""
    try:
        # Start uvicorn server
        uvicorn_process = start_uvicorn()
        
        # Wait for server to start
        time.sleep(2)
        
        # Start ngrok
        ngrok_process = start_ngrok_static()
        
        # Keep the script running
        try:
            while True:
                uvicorn_line = uvicorn_process.stdout.readline()
                if uvicorn_line:
                    print(f"[UVICORN] {uvicorn_line.strip()}")
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            uvicorn_process.terminate()
            ngrok_process.terminate()
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()