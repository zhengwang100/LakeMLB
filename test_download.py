"""测试所有 LakeMLB 数据集是否能正常下载和解压
下载的文件保存在 ./download_test/ 目录下，测试完成后请手动删除。
"""
import sys
import os
import os.path as osp
import shutil
import time

# 设置路径（rllm 位于 codes/lib/rllm）
PROJECT_ROOT = osp.abspath(osp.dirname(__file__))
LIB_ROOT = osp.join(PROJECT_ROOT, "codes", "lib")
if LIB_ROOT not in sys.path:
    sys.path.insert(0, LIB_ROOT)

from codes.lib.rllm.utils.download import download_url
from codes.lib.rllm.utils.extract import extract_zip

# 下载保存目录
SAVE_DIR = osp.join(PROJECT_ROOT, "download_test")
MAX_RETRIES = 3

# 所有数据集的 URL 和期望解压出的文件列表（与 codes/lib/rllm/datasets/lakemlb/*.py 中的 url 和 raw_filenames 保持一致）
DATASETS = {
    "mstraffic": {
        "url": "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/union_based/mstraffic.zip",
        "expected_files": [
            "maryland.csv", "seattle.csv", "mstraffic_da.csv", "mstraffic_fa.csv",
            "mask_maryland.pt", "mask_da.pt",
        ],
    },
    "ncbuilding": {
        "url": "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/union_based/ncbuilding.zip",
        "expected_files": [
            "newyork.csv", "chicago.csv", "ncbuilding_da.csv", "ncbuilding_fa.csv",
            "mask_newyork.pt", "mask_da.pt",
        ],
    },
    "gacars": {
        "url": "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/union_based/gacars.zip",
        "expected_files": [
            "german.csv", "australian.csv", "gacars_da.csv", "gacars_fa.csv",
            "mask_german.pt", "mask_da.pt",
        ],
    },
    "nnstocks": {
        "url": "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/join_based/nnstocks.zip",
        "expected_files": [
            "nnlist.csv", "nnwiki.csv", "nnstocks_da.csv", "nnstocks_fa.csv",
            "mask_nnlist.pt", "mask_da.pt",
        ],
    },
    "lhstocks": {
        "url": "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/join_based/lhstocks.zip",
        "expected_files": [
            "lhlist.csv", "lhwiki.csv", "lhstocks_da.csv", "lhstocks_fa.csv",
            "mask_lhlist.pt", "mask_da.pt",
        ],
    },
    "dsmusic": {
        "url": "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/join_based/dsmusic.zip",
        "expected_files": [
            "discogs.csv", "spotify.csv", "dsmusic_da.csv", "dsmusic_fa.csv",
            "mask_discogs.pt", "mask_da.pt",
        ],
    },
}


def test_dataset(name, info):
    """测试单个数据集的下载和解压，文件保存在 SAVE_DIR/<name>/ 下"""
    out_dir = osp.join(SAVE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{name}] 开始下载...")
    print(f"  URL: {info['url']}")

    # 带重试的下载
    zip_path = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            zip_path = download_url(info["url"], out_dir, f"{name}.zip")
            dl_time = time.time() - t0
            zip_size = os.path.getsize(zip_path) / (1024 * 1024)
            print(f"  下载完成: {zip_size:.1f} MB, 耗时 {dl_time:.1f}s")
            break
        except Exception as e:
            print(f"  第 {attempt}/{MAX_RETRIES} 次下载失败: {e}")
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"  等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  已达最大重试次数，跳过 ✗")
                return False

    try:
        # 解压
        extract_zip(zip_path, out_dir)
        os.remove(zip_path)

        # 检查文件
        extracted = set()
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                extracted.add(f)

        expected = set(info["expected_files"])
        missing = expected - extracted
        extra = extracted - expected

        if not missing:
            print(f"  解压验证: 全部 {len(expected)} 个文件均存在 ✓")
        else:
            print(f"  缺少文件: {missing} ✗")
        if extra:
            print(f"  额外文件: {extra}")

        print(f"  文件保存在: {out_dir}")
        return len(missing) == 0
    except Exception as e:
        print(f"  解压失败: {e} ✗")
        return False


if __name__ == "__main__":
    print("LakeMLB 数据集下载测试")
    print(f"共 {len(DATASETS)} 个数据集")
    print(f"文件将保存到: {SAVE_DIR}\n")

    results = {}
    for name, info in DATASETS.items():
        results[name] = test_dataset(name, info)

    # 汇总
    print(f"\n{'='*60}")
    print("测试汇总:")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name:12s} {status}")

    total_ok = sum(results.values())
    print(f"\n结果: {total_ok}/{len(results)} 通过")

    if total_ok < len(results):
        sys.exit(1)

