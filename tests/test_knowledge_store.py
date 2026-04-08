"""knowledge_store.py のテスト（SQLite、Ollama不要）"""
import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """テスト用の一時DBを使う"""
    db_path = str(tmp_path / "test_knowledge.db")
    monkeypatch.setenv("CHAT_DB_PATH", db_path)
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.knowledge_store as ks
    importlib.reload(ks)
    yield ks


class TestCreateWorkflow:
    def test_creates_workflow(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テストナレッジ", "品質", "田中", "admin1")
        assert isinstance(wf_id, str)
        assert len(wf_id) == 12

    def test_default_stage_is_draft(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("タイトル")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "draft"

    def test_stores_metadata(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("ノウハウ", "製造", "佐藤", "admin1")
        wf = ks.get_workflow(wf_id)
        assert wf["title"] == "ノウハウ"
        assert wf["category"] == "製造"
        assert wf["interviewee"] == "佐藤"
        assert wf["interviewer_id"] == "admin1"


class TestGetWorkflow:
    def test_returns_none_for_missing(self, tmp_db):
        ks = tmp_db
        assert ks.get_workflow("nonexistent") is None

    def test_returns_parsed_json_fields(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        wf = ks.get_workflow(wf_id)
        assert wf["interview_data"] == []
        assert wf["faq"] == []


class TestListWorkflows:
    def test_list_all(self, tmp_db):
        ks = tmp_db
        ks.create_workflow("A")
        ks.create_workflow("B")
        assert len(ks.list_workflows()) == 2

    def test_filter_by_stage(self, tmp_db):
        ks = tmp_db
        ks.create_workflow("A")
        ks.create_workflow("B")
        assert len(ks.list_workflows("draft")) == 2
        assert len(ks.list_workflows("summary")) == 0

    def test_order_by_updated_desc(self, tmp_db):
        ks = tmp_db
        id1 = ks.create_workflow("First")
        id2 = ks.create_workflow("Second")
        wfs = ks.list_workflows()
        assert wfs[0]["id"] == id2  # newer first


class TestUpdateInterviewData:
    def test_saves_qa_pairs(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        data = [{"q": "質問1", "a": "回答1"}, {"q": "質問2", "a": "回答2"}]
        assert ks.update_interview_data(wf_id, data)
        wf = ks.get_workflow(wf_id)
        assert len(wf["interview_data"]) == 2
        assert wf["interview_data"][0]["q"] == "質問1"

    def test_fails_for_non_draft(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.update_interview_data(wf_id, [{"q": "Q", "a": "A"}])
        ks.save_summary(wf_id, "要約", [{"q": "Q", "a": "A"}])
        # Now in summary stage
        assert not ks.update_interview_data(wf_id, [{"q": "new", "a": "new"}])


class TestSaveSummary:
    def test_saves_and_advances_to_summary(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.update_interview_data(wf_id, [{"q": "Q", "a": "A"}])
        faq = [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}]
        assert ks.save_summary(wf_id, "概要テキスト", faq)
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "summary"
        assert wf["summary"] == "概要テキスト"
        assert len(wf["faq"]) == 2

    def test_allows_regeneration_in_summary_stage(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.save_summary(wf_id, "旧要約", [{"q": "old", "a": "old"}])
        assert ks.save_summary(wf_id, "新要約", [{"q": "new", "a": "new"}])
        wf = ks.get_workflow(wf_id)
        assert wf["summary"] == "新要約"


class TestSubmitForReview:
    def test_advances_to_review(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.save_summary(wf_id, "要約", [{"q": "Q", "a": "A"}])
        assert ks.submit_for_review(wf_id)
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "review"

    def test_fails_from_draft(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        assert not ks.submit_for_review(wf_id)


class TestApproveWorkflow:
    def test_advances_to_published(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.save_summary(wf_id, "要約", [{"q": "Q", "a": "A"}])
        ks.submit_for_review(wf_id)
        assert ks.approve_workflow(wf_id, "reviewer1", "OK")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "published"
        assert wf["reviewer_id"] == "reviewer1"
        assert wf["review_notes"] == "OK"

    def test_fails_from_draft(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        assert not ks.approve_workflow(wf_id, "r1")


class TestRejectWorkflow:
    def test_returns_to_summary(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.save_summary(wf_id, "要約", [{"q": "Q", "a": "A"}])
        ks.submit_for_review(wf_id)
        assert ks.reject_workflow(wf_id, "reviewer1", "修正必要")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "summary"
        assert wf["review_notes"] == "修正必要"


class TestSetDocId:
    def test_records_doc_id(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        assert ks.set_doc_id(wf_id, "doc123")
        wf = ks.get_workflow(wf_id)
        assert wf["doc_id"] == "doc123"


class TestDeleteWorkflow:
    def test_deletes(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        assert ks.delete_workflow(wf_id)
        assert ks.get_workflow(wf_id) is None

    def test_returns_false_for_missing(self, tmp_db):
        ks = tmp_db
        assert not ks.delete_workflow("nonexistent")


class TestGetWorkflowStats:
    def test_counts_by_stage(self, tmp_db):
        ks = tmp_db
        ks.create_workflow("A")
        ks.create_workflow("B")
        wf_id = ks.create_workflow("C")
        ks.save_summary(wf_id, "要約", [{"q": "Q", "a": "A"}])
        stats = ks.get_workflow_stats()
        assert stats["draft"] == 2
        assert stats["summary"] == 1
        assert stats["total"] == 3

    def test_empty_stats(self, tmp_db):
        ks = tmp_db
        stats = ks.get_workflow_stats()
        assert stats["total"] == 0


class TestFullWorkflow:
    """draft → summary → review → published の全フロー"""

    def test_full_lifecycle(self, tmp_db):
        ks = tmp_db
        # 1. Create
        wf_id = ks.create_workflow("溶接ノウハウ", "製造", "山田太郎", "admin1")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "draft"

        # 2. Record interview
        qa = [
            {"q": "溶接で一番大事なことは？", "a": "温度管理と速度のバランスです"},
            {"q": "よくある失敗は？", "a": "速度が速すぎてビードが不安定になること"},
        ]
        ks.update_interview_data(wf_id, qa)

        # 3. Generate summary + FAQ
        faq = [
            {"q": "溶接で最も重要なポイントは何ですか？", "a": "温度管理と溶接速度のバランスが最も重要です。"},
            {"q": "よくある溶接の失敗例を教えてください。", "a": "溶接速度が速すぎるとビードが不安定になります。"},
        ]
        ks.save_summary(wf_id, "溶接作業では温度管理と速度バランスが重要。", faq)
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "summary"

        # 4. Submit for review
        ks.submit_for_review(wf_id)
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "review"

        # 5. Approve
        ks.approve_workflow(wf_id, "admin2", "内容確認済み")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "published"
        assert wf["reviewer_id"] == "admin2"

        # 6. Record doc_id after RAG publish
        ks.set_doc_id(wf_id, "doc_abc123")
        wf = ks.get_workflow(wf_id)
        assert wf["doc_id"] == "doc_abc123"

    def test_reject_and_retry(self, tmp_db):
        ks = tmp_db
        wf_id = ks.create_workflow("テスト")
        ks.save_summary(wf_id, "v1要約", [{"q": "Q", "a": "A"}])
        ks.submit_for_review(wf_id)

        # Reject
        ks.reject_workflow(wf_id, "admin2", "もう少し詳しく")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "summary"

        # Re-generate
        ks.save_summary(wf_id, "v2要約（詳細版）", [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}])

        # Submit again
        ks.submit_for_review(wf_id)
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "review"

        # Approve
        ks.approve_workflow(wf_id, "admin2", "OK")
        wf = ks.get_workflow(wf_id)
        assert wf["stage"] == "published"
