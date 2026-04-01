"""
フォルダ内のドキュメントを一括でベクトルDBに登録するスクリプト

使い方:
  python bulk_import.py <フォルダパス>
  python bulk_import.py <フォルダパス> --reset  (DB初期化してから登録)

例:
  python bulk_import.py ./docs
  python bulk_import.py "D:\社内文書\技術ドキュメント"
  python bulk_import.py ./docs --reset
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.parser import extract_text, SUPPORTED_EXTENSIONS
from app.vectorstore import add_document, get_stats, reset_db


async def bulk_import(folder: str, do_reset: bool = False):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"エラー: '{folder}' はフォルダではありません")
        sys.exit(1)

    # 対象ファイルを収集（サブフォルダも再帰的に探索）
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder_path.rglob(f"*{ext}"))
    files.sort()

    if not files:
        print(f"対象ファイルがありません（対応形式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}）")
        sys.exit(1)

    print(f"対象フォルダ: {folder_path.resolve()}")
    print(f"対象ファイル: {len(files)} 件")
    print(f"対応形式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print()

    # DB初期化
    if do_reset:
        reset_db()
        print("DB をリセットしました")
        print()

    # 登録
    total_chunks = 0
    success = 0
    errors: list[tuple[str, str]] = []
    start = time.time()

    for i, filepath in enumerate(files, 1):
        rel_path = filepath.relative_to(folder_path)
        try:
            data = filepath.read_bytes()
            text = extract_text(filepath.name, data)
            if not text.strip():
                errors.append((str(rel_path), "テキストが空でした"))
                print(f"  [{i}/{len(files)}] ⚠ {rel_path} — スキップ（テキストが空）")
                continue

            doc_id = filepath.stem[:8] + f"_{i}"
            num_chunks = await add_document(doc_id, str(rel_path), text)
            total_chunks += num_chunks
            success += 1
            print(f"  [{i}/{len(files)}] ✅ {rel_path} — {num_chunks} チャンク")
        except Exception as e:
            errors.append((str(rel_path), str(e)))
            print(f"  [{i}/{len(files)}] ❌ {rel_path} — {e}")

    elapsed = time.time() - start

    # サマリー
    print()
    print("=" * 50)
    print(f"完了: {elapsed:.1f} 秒")
    print(f"成功: {success}/{len(files)} ファイル")
    print(f"チャンク数: {total_chunks}")
    if errors:
        print(f"エラー: {len(errors)} 件")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print()
    stats = get_stats()
    print(f"DB合計チャンク数: {stats['total_chunks']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = sys.argv[1]
    do_reset = "--reset" in sys.argv
    asyncio.run(bulk_import(folder, do_reset))
