import json
from pathlib import Path

def main():
    data_dir = Path(__file__).parent.parent / "data"
    files = sorted([
        f.name for f in data_dir.glob("timing_data_*_*_*.csv")
    ])
    output = data_dir / "files.json"
    output.write_text(json.dumps(files, indent=2))
    print(f"Updated {output} with {len(files)} files")

if __name__ == "__main__":
    main()
