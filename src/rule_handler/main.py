from ragify import Ragify, cleanup_incomplete
from enum import Enum, auto
from pathlib import Path
from folder_watcher import watch, already_ragified
import _thread, time

class Status(Enum):
    RAGIFYING = auto()
    RAGIFIED = auto()

# Store folder status in memory
storage = {}
ragifying = False

def update_storage(folder, status):
    storage[folder] = status

def ragify_loop():

    global ragifying
    while True:
        folders = watch("/DandD/")  # Watch all edition folders
        to_ragify_folders = [f for f in folders if not already_ragified(f)]
        ragified_folders = [f for f in folders if already_ragified(f)]

        # Mark completed folders
        for folder in ragified_folders:
            update_storage(folder.name, Status.RAGIFIED)

        # Process incomplete folders
        for folder in to_ragify_folders:
            edition_path = Path(folder.name)
            complete_file = edition_path / ".ragify_complete"

            if complete_file.exists():
                # Already completed, mark as RAGIFIED
                update_storage(folder.name, Status.RAGIFIED)
                continue

            # Cleanup any leftover incomplete data
            cleanup_incomplete(edition_path)

            print(f"Starting Ragify on folder: {folder}")
            update_storage(folder.name, Status.RAGIFYING)

            try:
                Ragify(path=str(edition_path)).run()
                update_storage(folder.name, Status.RAGIFIED)
            except Exception as e:
                print(f"Error processing {folder}: {e}")

            ragifying = True

        # Only sleep if nothing to process
        if not ragifying:
            print("No folders to ragify")
            time.sleep(10)
            
        else:
            ragifying = False

if __name__ == "__main__":
    
    # folders = watch("/DandD/")  # Watch all edition folders
    # to_ragify_folders = [f for f in folders if not already_ragified(f)]
    # ragified_folders = [f for f in folders if already_ragified(f)]
    # print(to_ragify_folders)
    # for folder in to_ragify_folders:
    #     edition_path = Path(folder.name)
    #     cleanup_incomplete(edition_path)
    #     print(edition_path)
    #     try:
    #         Ragify(path=str(edition_path)).run()
    #     except Exception as e:
    #         print(f"Error processing {folder}: {e}")

    _thread.start_new_thread(ragify_loop, ())
    while True:
        time.sleep(.1)