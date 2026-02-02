#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


DEFAULT_ROOT = Path("/opt/Innovations/System")


def build_default_rollback_name(prefix: str = "pre_restore_backup") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def is_safe_member(member_name: str) -> bool:
    """
    ZIP内のファイル名が安全かチェック:
    - 絶対パス禁止
    - ドライブレター/UNCのような形（Windows系）も禁止
    - '..' を含むパストラバーサル禁止
    """
    # zipは常に / 区切りが基本
    name = member_name.replace("\\", "/")

    # 空・ディレクトリはOK（実ファイルのみ処理するが安全確認はしておく）
    if not name:
        return False

    # 絶対パス禁止
    if name.startswith("/"):
        return False

    # Windowsっぽい絶対パス禁止（例: C:/, \\server\share）
    if ":" in name.split("/")[0]:
        return False
    if name.startswith("//"):
        return False

    parts = [p for p in name.split("/") if p]
    if any(p == ".." for p in parts):
        return False

    return True


def list_zip(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        files = [n for n in names if not n.endswith("/")]
        print(f"zip: {zip_path}")
        print(f"entries: {len(names)} (files: {len(files)})")
        for n in files:
            ok = "OK" if is_safe_member(n) else "NG"
            print(f"[{ok}] {n}")
        # NGがあれば非0
        return 0 if all(is_safe_member(n) for n in names) else 2


def ensure_root(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ERROR: rootが存在しないかディレクトリではありません: {root}")


def safe_rel_from_member(member_name: str) -> Path:
    name = member_name.replace("\\", "/")
    if not is_safe_member(name):
        raise ValueError(f"危険なZIPエントリ名です: {member_name}")
    return Path(name)


def backup_current_files(target_files: list[Path], root: Path, backup_dir: Path) -> int:
    """
    リストアで上書きされる可能性のある現行ファイルを、相対パス構造を保って退避。
    存在しないファイルはスキップ。
    """
    count = 0
    for dst in target_files:
        if not dst.exists() or not dst.is_file():
            continue
        rel = dst.relative_to(root)
        out = backup_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, out)
        count += 1
    return count


def restore_zip(
    zip_path: Path,
    root: Path,
    dry_run: bool,
    make_pre_backup: bool,
    pre_backup_dir: Path | None,
    allow_new_files: bool,
) -> int:
    ensure_root(root)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        # 1) ZIPメンバ安全性チェック
        for m in members:
            if not is_safe_member(m.filename):
                print(f"ERROR: 危険なZIPエントリが含まれています: {m.filename}", file=sys.stderr)
                return 2

        # 2) 対象ファイル一覧（ファイルのみ）
        file_members = [m for m in members if not m.is_dir() and not m.filename.endswith("/")]
        if not file_members:
            print("ERROR: ZIP内に復元対象のファイルが見つかりません。", file=sys.stderr)
            return 2

        # 3) 復元先パスを作る（root + 相対）
        targets: list[tuple[zipfile.ZipInfo, Path]] = []
        for m in file_members:
            rel = safe_rel_from_member(m.filename)
            dst = (root / rel).resolve()
            # resolve後もroot配下であることを再確認（念には念）
            try:
                dst.relative_to(root.resolve())
            except ValueError:
                print(f"ERROR: 復元先がroot外です: {dst}", file=sys.stderr)
                return 2
            targets.append((m, dst))

        # allow_new_files=False の場合、存在しないファイルを拒否
        if not allow_new_files:
            missing = [dst for _, dst in targets if not dst.exists()]
            if missing:
                print("ERROR: 既存ファイルのみ許可(--no-new)ですが、存在しない復元先がありました:", file=sys.stderr)
                for p in missing:
                    print(f"  - {p}", file=sys.stderr)
                return 3

        # dry-run
        if dry_run:
            print(f"restore root: {root}")
            print(f"zip: {zip_path}")
            print(f"files: {len(targets)}")
            for m, dst in targets:
                action = "overwrite" if dst.exists() else "create"
                print(f"{action}: {dst}  <-  {m.filename}")
            return 0

        # 4) 上書き前バックアップ（任意）
        if make_pre_backup:
            if pre_backup_dir is None:
                raise SystemExit("INTERNAL ERROR: pre_backup_dir is None")
            pre_backup_dir.mkdir(parents=True, exist_ok=True)
            dst_paths = [dst for _, dst in targets]
            backed = backup_current_files(dst_paths, root, pre_backup_dir)
            print(f"OK: pre-backup {backed} files -> {pre_backup_dir}")

        # 5) ZIPを一時領域へ展開してから、目的地へcopy2
        with tempfile.TemporaryDirectory(prefix="cfg_restore_") as tmpdir:
            tmp = Path(tmpdir)
            # まず展開
            zf.extractall(path=tmp)

            restored = 0
            for m, dst in targets:
                rel = safe_rel_from_member(m.filename)
                src = (tmp / rel).resolve()
                if not src.exists() or not src.is_file():
                    print(f"ERROR: 展開後にファイルが見つかりません: {src}", file=sys.stderr)
                    return 4

                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored += 1

        print(f"OK: restored {restored} files into {root}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backup.py が作成したZIP（root配下の相対パス構造）を、指定rootへ安全にリストアします。"
    )

    parser.add_argument("zip", help="リストアするZIPファイルパス")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"復元先root（デフォルト: {DEFAULT_ROOT}）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="ZIPの中身を表示して終了（安全性チェックも行う）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には復元せず、何が上書き/作成されるか表示する",
    )

    # 上書き前バックアップ
    parser.add_argument(
        "--pre-backup",
        action="store_true",
        help="上書き前に、現行ファイルを退避バックアップする",
    )
    parser.add_argument(
        "--pre-backup-dir",
        default="",
        help="退避バックアップ先ディレクトリ（省略時: ./backups/pre_restore_backup_YYYYmmdd_HHMMSS/）",
    )

    # 新規ファイル作成を禁止したい場合
    parser.add_argument(
        "--no-new",
        action="store_true",
        help="ZIP内にあるが現行rootに存在しないファイルは復元しない（エラーにする）",
    )

    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()

    if not zip_path.exists() or not zip_path.is_file():
        print(f"ERROR: ZIPが存在しません: {zip_path}", file=sys.stderr)
        return 2

    if args.list:
        return list_zip(zip_path)

    pre_backup_dir: Path | None = None
    if args.pre_backup:
        if args.pre_backup_dir.strip():
            pre_backup_dir = Path(args.pre_backup_dir).expanduser().resolve()
        else:
            pre_backup_dir = Path("./backups") / build_default_rollback_name()

    return restore_zip(
        zip_path=zip_path,
        root=root,
        dry_run=args.dry_run,
        make_pre_backup=args.pre_backup,
        pre_backup_dir=pre_backup_dir,
        allow_new_files=not args.no_new,
    )


if __name__ == "__main__":
    raise SystemExit(main())
