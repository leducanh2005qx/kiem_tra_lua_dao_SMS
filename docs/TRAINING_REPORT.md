# SCAM model training report

**BLOCKED** — chưa có đủ dữ liệu `SCAM`/`NOT_SCAM` đã được người xác minh.

- Candidate trong queue: **2991**
- Mẫu verified: **0**
- Mẫu `NEEDS_REVIEW`: **0**
- Lý do chặn: `fewer_than_40_verified_rows, fewer_than_20_verified_SCAM, fewer_than_20_verified_NOT_SCAM, fewer_than_6_verified_groups`

`results/error_analysis.csv` chưa được thay thế: file hiện tại là error analysis của model nhị phân legacy, không phải model SCAM.

Nhãn nhị phân cũ `0/1` không được chuyển thành nhãn scam. Chạy lại sau khi annotation và xác minh dữ liệu:

```powershell
python scripts/train_scam_model.py audit
python scripts/train_scam_model.py train
```
