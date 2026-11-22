from src.storage import (
    activity_summary,
    add_analysis,
    add_feedback,
    clear_history,
    delete_analysis,
    feedback_csv,
    get_analysis,
    history_csv,
    initialize_database,
    list_feedback,
    list_history,
    update_feedback_status,
)


def test_empty_database_dashboard_has_zero_metrics(tmp_path):
    db = tmp_path / "history.db"
    initialize_database(db)
    assert activity_summary(db_path=db) == {
        "total": 0,
        "valid": 0,
        "suspicious": 0,
        "suspicious_rate": 0.0,
        "feedback_total": 0,
        "feedback_pending": 0,
    }


def test_analysis_privacy_feedback_filters_delete_and_export(tmp_path):
    db = tmp_path / "history.db"
    analysis_id = add_analysis("svm", 1, 0.8, "decision_margin", "secret SMS", db_path=db)
    saved_id = add_analysis("nb_baseline", 0, 0.2, "posterior_label_1", "keep this", save_content=True, db_path=db)

    assert get_analysis(analysis_id, db)["message_text"] is None
    assert get_analysis(saved_id, db)["message_text"] == "keep this"
    rows, total = list_history(search="keep", db_path=db)
    assert total == 1 and rows[0]["id"] == saved_id

    feedback_id = add_feedback(analysis_id, "incorrect", 0, "cần xem lại", db_path=db)
    feedback, total_feedback = list_feedback(status="pending", db_path=db)
    assert total_feedback == 1 and feedback[0]["id"] == feedback_id
    assert update_feedback_status(feedback_id, "reviewed", db)
    assert list_feedback(status="reviewed", db_path=db)[1] == 1
    assert b"secret SMS" not in history_csv(list_history(db_path=db)[0])
    assert "cần xem lại".encode("utf-8") in feedback_csv(list_feedback(db_path=db)[0])

    assert delete_analysis(analysis_id, db)
    clear_history(db)
    assert activity_summary(db_path=db)["total"] == 0


def test_hybrid_analysis_feedback_fields_and_review_states(tmp_path):
    db = tmp_path / "hybrid.db"
    analysis_id = add_analysis(
        "svm",
        0,
        0.2,
        "decision_margin",
        "Cung cấp OTP",
        save_content=True,
        db_path=db,
        hybrid_result={
            "final_label": "PHISHING",
            "risk_level": "cao",
            "rule_override": True,
            "needs_review": False,
            "reasons": ["Yêu cầu OTP"],
            "signals": [{"code": "PHISHING_CREDENTIAL_REQUEST"}],
            "urls": [],
            "evidence": ["Cung cấp OTP"],
        },
    )
    saved = get_analysis(analysis_id, db)
    assert saved["hybrid_label"] == "PHISHING"
    assert saved["rule_override"] == 1
    feedback_id = add_feedback(analysis_id, "incorrect", 2, "Sai nhãn", db, suggested_label_code="PHISHING", feedback_reason="Có yêu cầu OTP")
    rows, total = list_feedback(status="pending_review", db_path=db)
    assert total == 1 and rows[0]["suggested_label_code"] == "PHISHING"
    assert update_feedback_status(feedback_id, "verified", db)
    assert list_feedback(status="verified", db_path=db)[1] == 1
