from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import cast


class FolderWatcherHandler(FileSystemEventHandler):
    def on_created(self, event):
        folder_path = Path(cast(str, event.src_path))
        if folder_path.is_file():
            folder_path = folder_path.parent

        print(f"Detected new folder: {folder_path}")

        if to_be_ragified(folder_path):
            print(f"Ragifying folder: {folder_path}")
            # Ragify(folder_path).run()
        elif already_ragified(folder_path):
            print(f"Folder already ragified: {folder_path}")


def watch(path) -> list[Path]:
    """Watches a folder and returns a list of all folders in it."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return [d for d in p.iterdir() if d.is_dir()]


def contains_file(folder: Path, filename: str) -> bool:
    """Checks if a folder contains a specific file."""
    return any(f.is_file() and f.name == filename for f in folder.iterdir())


def to_be_ragified(folder: Path) -> bool:
    """
    A folder should be ragified if:
      - It is inside ./DandD/
      - It contains ruleset.md
      - It does NOT contain .ragify_complete
    """
    if folder.parent.name != "DandD":
        # Not a version folder inside DandD
        return False

    has_ruleset = contains_file(folder, "ruleset.md")
    is_complete = contains_file(folder, ".ragify_complete")

    print(f"[Check] Folder: {folder}, has_ruleset: {has_ruleset}, ragify_complete: {is_complete}")
    return has_ruleset and not is_complete


def already_ragified(folder: Path) -> bool:
    """A folder is considered ragified if it contains .ragify_complete."""
    return contains_file(folder, ".ragify_complete")


if __name__ == "__main__":
    observer = Observer()
    path_to_watch = "~/project-nigle/DandD"
    observer.schedule(FolderWatcherHandler(), path=path_to_watch, recursive=True)
    observer.start()
    print(f"Watching folder: {path_to_watch}")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
