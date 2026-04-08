"""ページ画像ストアのテスト"""

import os
import pytest
from app.page_image_store import (
    save_page_image,
    get_page_images,
    get_all_page_images,
    delete_page_images,
    count_page_images,
)


class TestPageImageStore:
    def test_save_and_get(self):
        save_page_image("test.pdf", 1, "base64data1", 800, 600)
        save_page_image("test.pdf", 2, "base64data2", 800, 600)
        result = get_page_images("test.pdf", [1, 2])
        assert len(result) == 2
        assert result[0]["page"] == 1
        assert result[0]["image_b64"] == "base64data1"
        assert result[1]["page"] == 2
        # クリーンアップ
        delete_page_images("test.pdf")

    def test_get_all(self):
        save_page_image("all_test.pdf", 1, "img1", 100, 100)
        save_page_image("all_test.pdf", 2, "img2", 100, 100)
        save_page_image("all_test.pdf", 3, "img3", 100, 100)
        result = get_all_page_images("all_test.pdf")
        assert len(result) == 3
        delete_page_images("all_test.pdf")

    def test_count(self):
        save_page_image("count_test.pdf", 1, "c1", 0, 0)
        save_page_image("count_test.pdf", 2, "c2", 0, 0)
        assert count_page_images("count_test.pdf") == 2
        delete_page_images("count_test.pdf")

    def test_delete(self):
        save_page_image("del_test.pdf", 1, "d1", 0, 0)
        assert count_page_images("del_test.pdf") == 1
        deleted = delete_page_images("del_test.pdf")
        assert deleted == 1
        assert count_page_images("del_test.pdf") == 0

    def test_upsert(self):
        save_page_image("upsert.pdf", 1, "old_data", 100, 100)
        save_page_image("upsert.pdf", 1, "new_data", 200, 200)
        result = get_page_images("upsert.pdf", [1])
        assert len(result) == 1
        assert result[0]["image_b64"] == "new_data"
        assert result[0]["width"] == 200
        delete_page_images("upsert.pdf")

    def test_get_empty(self):
        result = get_page_images("nonexistent.pdf", [1, 2])
        assert result == []

    def test_get_all_empty(self):
        result = get_all_page_images("nonexistent.pdf")
        assert result == []
