from src.data_utils import load_sms_csv, remove_exact_test_overlap


def test_clean_duplicate_and_overlap(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("message,label\nXin chào,0\nXin chào,0\nNhận quà ngay,1\n", encoding="utf-8-sig")
    data, stats = load_sms_csv(train)
    assert len(data) == 2
    assert stats["duplicate_rows_removed"] == 1
    filtered, n_messages, n_rows = remove_exact_test_overlap(data, data.iloc[:1])
    assert n_messages == 1 and n_rows == 1
    assert len(filtered) == 1
