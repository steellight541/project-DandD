import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os
import signal
import threading

WATCH_PATH = "/DandD"
IGNORE_PATHS = ["split_ruleset", "rag_storage", "logs", ".ragify_complete"]

class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.busy = False
        self.lock = threading.Lock()
        self.restart_pending = False
        self.start_process()

    def start_process(self):
        """Start main.py if not already running."""
        if not self.busy:
            print("▶ Starting main.py...")
            self.busy = True
            self.process = subprocess.Popen(["python", "main.py"])
            threading.Thread(target=self.wait_for_process).start()

    def wait_for_process(self):
        """Wait for the process to finish and handle pending restarts."""
        self.process.wait()
        with self.lock:
            self.busy = False
            self.process = None

    def on_any_event(self, event):
        # Ignore directories and ignored paths
        if event.is_directory:
            return
        if any(p in event.src_path for p in IGNORE_PATHS):
            return
        print("🔄 Change detected:", event.src_path)
        self.start_process()


if __name__ == "__main__":
    event_handler = ReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_PATH, recursive=True)
    observer.start()

    print(f"👀 Watching for changes in {WATCH_PATH}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if event_handler.process:
            os.kill(event_handler.process.pid, signal.SIGTERM)

    observer.join()
