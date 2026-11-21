"""Download official open SMS corpus. Run: python download_data.py"""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from src.config import RAW_DIR, DATA_URLS
from src.data_utils import sha256_file


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"source": "trannguyenthaituan/vietnamese_sms_dataset", "download_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for filename, url in DATA_URLS.items():
        target = RAW_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"Có sẵn: {target} (không tải đè)")
        else:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 SMS-NLP-Coursework/1.0"})
            try:
                with urlopen(req, timeout=45) as resp:
                    data = resp.read()
                if not data or not data.lstrip(b"\xef\xbb\xbf").startswith(b"message,label"):
                    raise ValueError("Dữ liệu tải về không đúng header message,label")
                target.write_bytes(data)
                print(f"Đã tải: {target}, {len(data):,} bytes")
            except Exception as exc:
                print(f"Lỗi tải {filename}: {exc}")
                print(f"Bạn có thể tải thủ công từ {url} rồi lưu vào {target}")
                raise SystemExit(1) from exc
        metadata["files"][filename] = {"url": url, "sha256": sha256_file(target), "bytes": target.stat().st_size}
    (RAW_DIR / "source_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dữ liệu đã sẵn sàng. Chạy: python train.py")


if __name__ == "__main__":
    main()
