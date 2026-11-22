# Training Readiness Audit

**BLOCKED – chưa đủ dữ liệu SCAM/NOT_SCAM đã được người xác minh.**

- Candidate trong queue: **2991**
- Verified trước loại trùng: **0**
- Verified sau loại trùng: **0**
- SCAM verified: **0**
- NOT_SCAM verified: **0**
- NEEDS_REVIEW: **0**
- Annotator agreement: **chưa có hai bộ nhãn hợp lệ**
- Lý do chặn: `fewer_than_40_verified_rows, fewer_than_20_verified_SCAM, fewer_than_20_verified_NOT_SCAM, fewer_than_6_verified_groups`

Nhãn nhị phân cũ `0/1` chỉ mô tả hợp lệ/rác-nghi vấn và không được tự động chuyển thành nhãn scam.

Gán nhãn theo `docs/SCAM_LABELING_GUIDE.md`, sau đó chạy lại `python scripts/train_scam_model.py audit`.
