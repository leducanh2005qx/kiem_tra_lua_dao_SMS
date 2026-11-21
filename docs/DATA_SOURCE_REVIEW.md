# Data Source Review – Phase 0

**Ngày kiểm tra:** 2026-11-22  
**Nguyên tắc:** không tải/gộp dữ liệu trong audit; không xem các nguồn có quan hệ nguồn gốc là independent.

## Bảng tổng hợp

| Nguồn | Cấu trúc quan sát được | Nhãn | Provenance/synthetic | License hiển thị | Đánh giá |
|---|---|---|---|---|---|
| `trannguyenthaituan/vietnamese_sms_dataset` | CSV, full khoảng 2,99k; `message_id,date,message,label` | `0/1` | Vietnamese SMS; raw CSV không có per-row synthetic/provenance flag | CC BY 4.0 | Nguồn hiện tại/local baseline; không đủ cho 4 lớp |
| `tocong282/vietnamese-fraud-detection` | Repository có data/processed, notebooks, PhoBERT pipeline | `legitimate`, `phishing`, `scam_loan`, `promotion` theo README | README nói xây từ nguồn Vietnamese SMS ở hàng trên và có synthetic template augmentation cho minority | Chưa xác minh được license dữ liệu/model trong repository page | Derivative/research reference; không independent, không gộp trực tiếp |
| `wikiepeidia/vnphish-dataset` | JSON; train hiển thị khoảng 1,66k; `text,label,risk_tier,suspicious_spans,xai_explanation,source,seed_id` | 4 label values theo dataset card, thiên về bank impersonation/phishing | Dataset card gắn `Synthetic`; source có `synthetic_claude`/`synthetic_openai_compatible` | CC BY-NC 4.0 | Chỉ ứng viên nghiên cứu sau license/duplicate/mapping audit; không dùng production/commercial mặc định |

## 1. Nguồn Hugging Face hiện tại

URL: <https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset>

Trang dataset hiển thị tiếng Việt, task text classification, CSV, schema `message_id`, `date`, `message`, `label`, khoảng 2,99k dòng và license CC BY 4.0. Local `data/raw/train.csv` + `test.csv` có đúng 2.991 dòng, cùng kiểu schema rút gọn `message,label`, và `source_metadata.json` lưu SHA-256/URL của nguồn.

Kết luận: đây là một nguồn gốc nhị phân duy nhất, không phải ba nguồn độc lập. Có thể giữ làm binary baseline/reference và làm nguồn ứng viên để human re-annotation, nhưng không được gán bốn lớp bằng keyword hay dùng lại test cũ như four-class test.

## 2. Repository `tocong282/vietnamese-fraud-detection`

URL: <https://github.com/tocong282/vietnamese-fraud-detection>

README công bố bốn nhóm `legitimate`, `phishing`, `scam_loan`, `promotion`, báo cáo PhoBERT và nói dữ liệu được xây từ dataset Vietnamese SMS ở nguồn 1. README cũng nêu training set kết hợp collected messages với synthetic template augmentation cho minority classes, và synthetic examples được đánh dấu bởi data-generation script.

Đây là bằng chứng quan trọng rằng schema bốn lớp có thể tồn tại ở một pipeline nghiên cứu, nhưng không chứng minh nhãn bốn lớp là ground truth độc lập. README còn mô tả “binary-to-four-class heuristic labels” trong phần rebuild data; do đó phải coi nhãn là derived/heuristic cho đến khi kiểm tra toàn bộ file annotation và script.

Quyết định: không tải/gộp vào project ở Phase 0. Nếu dùng sau này, cần xác minh commit, file dữ liệu gốc, license dataset/model, tỷ lệ synthetic và parent/source_id; phải deduplicate với nguồn 1 trước khi split.

## 3. Dataset `wikiepeidia/vnphish-dataset`

URL: <https://huggingface.co/datasets/wikiepeidia/vnphish-dataset>

Dataset card hiển thị JSON, tiếng Việt, khoảng 1,66k train, bốn giá trị label, các trường `risk_tier`, `suspicious_spans`, `xai_explanation`, `source`, `seed_id`; dataset được gắn tag `Synthetic` và license CC BY-NC 4.0. Ví dụ hiển thị các nguồn tạo như `synthetic_claude` và `synthetic_openai_compatible`.

Schema này không khớp trực tiếp với quy chuẩn `LEGITIMATE/PROMOTION/PHISHING/FINANCIAL_SCAM`; các label cần đọc đầy đủ bằng manifest trước khi mapping. `source`/`seed_id` hữu ích để group split, nhưng cũng cho thấy các mẫu có thể có quan hệ sinh từ cùng seed/template.

Quyết định: không dùng làm production data, không dùng làm independent test, không gộp khi chưa có đánh giá license NC, synthetic bias, source dependency và duplicate với local/derived source. Có thể dùng làm nguồn nghiên cứu phishing phụ trợ nếu được phê duyệt pháp lý và tách split rõ ràng.

## 4. Khả năng kết hợp

Không được nối ba nguồn theo hàng rồi train. Quy trình tối thiểu nếu được phê duyệt:

1. Tạo `source_manifest` ghi URL, commit/revision, license, ngày tải, schema và SHA-256.
2. Chuẩn hóa về một schema canonical nhưng giữ `source_label` gốc và `source_id`.
3. Map nhãn bằng bảng được người dùng/annotator duyệt; không dùng keyword mapping tự động.
4. Giữ `synthetic`, `derived`, `real/unknown` thành các trường riêng.
5. Deduplicate exact normalized và cluster near-duplicate xuyên nguồn.
6. Group split theo source/parent/seed/template để tránh leakage.
7. Chỉ dùng nguồn license phù hợp với mục đích; CC BY-NC không được mặc định dùng cho sản phẩm thương mại.

## 5. Khuyến nghị sử dụng

- **Train bốn lớp:** chỉ dùng dòng đã được con người xác minh, có provenance và `annotation_status=certain`.
- **Validation:** có thể dùng collected/derived đã audit nhưng phải tách group khỏi train.
- **Test:** ưu tiên dữ liệu thật, nguồn khác hoặc thời gian khác; không dùng synthetic cùng template với train.
- **Synthetic:** báo cáo riêng; không gộp metric synthetic và real thành một con số nếu chưa có thiết kế đánh giá phù hợp.

## 6. Trạng thái phê duyệt

| Nguồn | Được tải trong Phase 0 | Được gộp | Được dùng train 4 lớp | Được dùng test |
|---|---|---|---|---|
| Local/HF binary | Không tải lại | Chưa | Chưa, cần re-annotation | Chỉ binary baseline hiện tại |
| GitHub derivative | Không | Không | Chưa xác minh | Không |
| vnphish synthetic NC | Không | Không | Chưa, cần license/mapping audit | Không |

Không có external dataset nào được thêm vào repository hoặc huấn luyện trong Phase 0.
