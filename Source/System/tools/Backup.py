#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import os
import pwd
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/opt/Innovations/System")
DEFAULT_PATTERNS = ("*.ini", "*.xml", "*.bin", "*.db")


def get_real_user_home(prefer_username: str | None = None) -> tuple[str, int, int, Path]:
    """
    sudo実行でも「元の通常ユーザー」のホームを特定する。
    戻り値: (username, uid, gid, home_path)
    """
    # 1) sudoでの実行なら SUDO_USER / SUDO_UID / SUDO_GID を優先
    sudo_user = os.environ.get("SUDO_USER")
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")

    if prefer_username:
        pw = pwd.getpwnam(prefer_username)
        return pw.pw_name, pw.pw_uid, pw.pw_gid, Path(pw.pw_dir)

    if sudo_user and sudo_uid and sudo_gid:
        uid = int(sudo_uid)
        gid = int(sudo_gid)
        pw = pwd.getpwuid(uid)
        return sudo_user, uid, gid, Path(pw.pw_dir)

    # 2) sudoじゃない通常実行は現在ユーザー
    uid = os.getuid()
    pw = pwd.getpwuid(uid)
    return pw.pw_name, pw.pw_uid, pw.pw_gid, Path(pw.pw_dir)


def iter_matches(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                found.add(p)
    return sorted(found)


def safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as e:
        raise ValueError(f"対象ファイルがroot外です: {path}") from e


def copy_to_staging(files: list[Path], root: Path, staging: Path) -> int:
    count = 0
    for src in files:
        rel = safe_relpath(src, root)
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def make_zip_from_dir(staging: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in staging.rglob("*"):
            if p.is_file():
                arcname = p.relative_to(staging)
                zf.write(p, arcname.as_posix())


def build_default_zip_name(prefix: str = "config_backup") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.zip"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="設定ファイル(ini/xml/bin/db)を一時領域に集めてZIP化し、通常ユーザーのホームへ保存します（sudo対応）。"
    )

    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"探索対象ルート（デフォルト: {DEFAULT_ROOT}）",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=list(DEFAULT_PATTERNS),
        help=f"対象拡張子パターン（デフォルト: {' '.join(DEFAULT_PATTERNS)}）",
    )
    parser.add_argument(
        "--name",
        default="",
        help="出力ZIP名（省略時: config_backup_YYYYmmdd_HHMMSS.zip）",
    )
    parser.add_argument(
        "--output-dir",
        default="AUTO",
        help="出力先。AUTOなら通常ユーザーのホーム（sudo時もSUDO_USERのホーム）。",
    )
    parser.add_argument(
        "--as-user",
        default="",
        help="出力先ユーザーを明示（例: --as-user tomoya）。指定時はそのユーザーのホームに出力。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せず対象ファイル一覧だけ表示",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    zip_name = args.name.strip() or build_default_zip_name()

    # 出力先ユーザー特定
    prefer_user = args.as_user.strip() or None
    username, uid, gid, real_home = get_real_user_home(prefer_user)

    if args.output_dir == "AUTO":
        out_dir = real_home
    else:
        out_dir = Path(args.output_dir).expanduser().resolve()

    # root存在チェック
    if not root.exists() or not root.is_dir():
        print(f"ERROR: rootが存在しないかディレクトリではありません: {root}", file=sys.stderr)
        return 2

    # 読み取り権限チェック（ざっくり）
    if not os.access(root, os.R_OK):
        print(f"WARNING: {root} が読めない可能性があります。sudoで実行してください。", file=sys.stderr)

    matches = iter_matches(root, args.patterns)

    if args.dry_run:
        print(f"scan root: {root}")
        print(f"patterns: {args.patterns}")
        print(f"hit: {len(matches)} files")
        print(f"output user: {username} (uid={uid})")
        print(f"output dir: {out_dir}")
        for p in matches:
            print(p)
        return 0

    if len(matches) == 0:
        print(f"対象ファイルが見つかりませんでした: root={root} patterns={args.patterns}")
        return 1

    # ZIPは一時領域で生成し、最後に移動
    with tempfile.TemporaryDirectory(prefix="cfg_backup_") as tmpdir:
        tmp = Path(tmpdir)
        staging = tmp / "staging"
        staging.mkdir(parents=True, exist_ok=True)

        copied = copy_to_staging(matches, root, staging)

        tmp_zip = tmp / zip_name
        make_zip_from_dir(staging, tmp_zip)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / zip_name

        if out_path.exists():
            out_path.unlink()

        shutil.move(str(tmp_zip), str(out_path))

    # sudoで作成されたZIPの所有者を元ユーザーへ戻す
    try:
        os.chown(out_path, uid, gid)
    except PermissionError:
        # sudo無し等でchownできないケースは無視（通常は問題なし）
        pass

    print(f"OK: {copied} files -> {out_path}")
    print(f"owner: {username}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
