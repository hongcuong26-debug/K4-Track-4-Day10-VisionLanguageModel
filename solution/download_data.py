"""Download the public Kaggle dataset without putting API credentials in code."""
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://www.kaggle.com/api/v1/datasets/download/"
       "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes")


def main():
    archive = Path("work/road_damage_dataset.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as file:
            total = 0
            while chunk := response.read(1024*1024):
                file.write(chunk)
                total += len(chunk)
                if total % (20*1024*1024) == 0:
                    print(f"Downloaded {total/1024**2:.0f} MiB", flush=True)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError("Kaggle did not return a ZIP; authenticate through Colab Secrets instead")
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--zip", str(archive)], check=True)


if __name__ == "__main__":
    main()
