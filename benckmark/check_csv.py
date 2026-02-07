import os
import pandas as pd
import chardet

root_dir = os.path.dirname(os.path.abspath(__file__))

ENCODINGS_TO_TRY = ["utf-8", "gbk", "gb2312", "latin-1", "iso-8859-1", "cp1252", "big5", "shift_jis"]


def detect_and_read(filepath):
    """尝试多种编码读取CSV，若非UTF-8则自动转换并覆盖保存为UTF-8"""
    # 先尝试 utf-8
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        return df, "utf-8", False
    except (UnicodeDecodeError, Exception):
        pass

    # 用 chardet 检测编码
    with open(filepath, "rb") as f:
        raw = f.read()
    detected = chardet.detect(raw)
    det_enc = detected.get("encoding")

    # 尝试 chardet 检测到的编码
    if det_enc:
        try:
            df = pd.read_csv(filepath, encoding=det_enc)
            # 转存为 UTF-8
            df.to_csv(filepath, index=False, encoding="utf-8")
            return df, det_enc, True
        except Exception:
            pass

    # 逐一尝试候选编码
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            df.to_csv(filepath, index=False, encoding="utf-8")
            return df, enc, True
        except Exception:
            continue

    raise RuntimeError("所有编码均无法读取")


print(f"{'文件路径':<55} {'样本数':>8} {'列数':>6}  {'原编码':<12} {'已转换'}")
print("-" * 95)

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in sorted(filenames):
        if filename.endswith(".csv"):
            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, root_dir)
            try:
                df, orig_enc, converted = detect_and_read(filepath)
                status = "✔ 已转为UTF-8" if converted else "原始UTF-8"
                print(f"{rel_path:<55} {df.shape[0]:>8} {df.shape[1]:>6}  {orig_enc:<12} {status}")
            except Exception as e:
                print(f"{rel_path:<55} 读取失败: {e}")

print("-" * 95)
print("检查完毕。")

