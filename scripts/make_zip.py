# scripts/make_zip.py
import argparse
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser("Zip the corpus folder")
    ap.add_argument("--folder", type=Path, default=Path("corpus"))
    ap.add_argument("--out", type=Path, default=Path("corpus.zip"))
    ap.add_argument("--fresh", action="store_true", help="Remove existing zip first")
    args = ap.parse_args()

    if args.fresh and args.out.exists():
        args.out.unlink()

    if not args.folder.exists():
        print(f"{args.folder} does not exist")
        return

    shutil.make_archive(args.out.with_suffix(""), "zip", root_dir=args.folder)
    print(f"[OK] Wrote {args.out}")

if __name__ == "__main__":
    main()
