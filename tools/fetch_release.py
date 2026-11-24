# -*- coding: utf-8 -*-
"""Fetch weights / portable runtime from GitHub Release assets (no pip needed).

usage:
  python tools/fetch_release.py --list
  python tools/fetch_release.py --only weights
  python tools/fetch_release.py --only runtime
  python tools/fetch_release.py --only all            # weights + runtime

It reads tools/release.json:  {"repo": "OWNER/REPO", "tag": "v6.2", "assets": [{...}]}
Volumes (name.zip.001, name.zip.002 ...) are downloaded, merged, sha256-verified
and extracted with 7za.exe / 7z.exe (bundled at tools/bin/7za.exe if present).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
MANIFEST = TOOLS / "release.json"
DL = ROOT / "downloads"


def load_manifest():
    if not MANIFEST.exists():
        sys.exit(f"[fatal] {MANIFEST} not found")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dst, expect_size=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if expect_size and have == expect_size:
        part.replace(dst)
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": "ddsp-fetch/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
        total = r.headers.get("Content-Length")
        total = int(total) + have if total else None
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            have += len(buf)
            if total:
                pct = have * 100 // total
                print(f"\r    {dst.name}: {human(have)}/{human(total)} ({pct}%)", end="")
    print()
    part.replace(dst)
    return dst


def find_7z():
    for name in ("7za.exe", "7z.exe", "7zr.exe"):
        p = TOOLS / "bin" / name
        if p.exists():
            return str(p)
    for name in ("7za", "7z", "7zz"):
        p = shutil.which(name)
        if p:
            return p
    return None


def merge_volumes(vol_files, out_path):
    """concatenate name.zip.001/.002/... into one file"""
    print(f"  merging {len(vol_files)} volume(s) -> {out_path.name}")
    with open(out_path, "wb") as out:
        for v in vol_files:
            with open(v, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
    return out_path


def extract(archive, dest):
    z = find_7z()
    if not z:
        sys.exit("  [fatal] 7za.exe not found -> put 7za.exe into tools\\bin\\ "
                 "or extract manually: " + str(archive))
    print(f"  extracting {archive.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [z, "x", "-y", f"-o{dest}", str(archive)]
    subprocess.check_call(cmd)


def do_group(asset):
    name = asset["name"]
    parts = asset.get("parts") or [name]
    print(f"\n=== {name}  ({human(asset.get('size', 0))})")
    vol_paths = []
    for p in parts:
        dst = DL / p["name"]
        if dst.exists() and dst.stat().st_size == p.get("size", dst.stat().st_size):
            print(f"  cached {dst.name} ({human(dst.stat().st_size)})")
        else:
            print(f"  downloading {p['url']}")
            download(p["url"], dst, p.get("size"))
        if p.get("sha256"):
            got = sha256_of(dst)
            if got != p["sha256"]:
                sys.exit(f"  [fatal] {dst.name} sha256 mismatch\n   want {p['sha256']}\n   got  {got}")
            print(f"  {dst.name} sha256 OK")
        vol_paths.append(dst)

    merged = DL / name
    if len(vol_paths) > 1:
        merge_volumes(vol_paths, merged)
    else:
        shutil.copyfile(vol_paths[0], merged)

    if asset.get("sha256"):
        print("  verifying merged sha256 ...")
        got = sha256_of(merged)
        if got != asset["sha256"]:
            sys.exit(f"  [fatal] sha256 mismatch\n   want {asset['sha256']}\n   got  {got}")
        print("  sha256 OK")

    if asset.get("extract", True):
        extract(merged, ROOT)
        print(f"  extracted into {ROOT}")
        for v in vol_paths:
            v.unlink(missing_ok=True)
        merged.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "weights", "runtime"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    man = load_manifest()
    groups = man.get("assets", [])
    if a.list:
        print(f"repo: {man.get('repo')}   tag: {man.get('tag')}")
        for g in groups:
            print(f"  {g.get('group', '?'):8s} {g['name']:50s} {human(g.get('size', 0))}")
        return
    todo = [g for g in groups if a.only in ("all", g.get("group"))]
    if not todo:
        sys.exit(f"no asset group '{a.only}' in manifest")
    for g in todo:
        do_group(g)
    print("\nall done.")


if __name__ == "__main__":
    main()
