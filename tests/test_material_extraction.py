from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient

from app.main import app


def test_extract_text_material() -> None:
    response = TestClient(app).post(
        "/api/v1/profile/materials/extract",
        files={"file": ("experience.md", "我负责用户访谈，并根据反馈修改原型。".encode("utf-8"), "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "我负责用户访谈，并根据反馈修改原型。"
    assert response.json()["truncated"] is False


def test_extract_docx_paragraphs_and_tables() -> None:
    document = Document()
    document.add_paragraph("项目经历：完成需求分析。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "行动"
    table.cell(0, 1).text = "访谈用户"
    buffer = BytesIO()
    document.save(buffer)

    response = TestClient(app).post(
        "/api/v1/profile/materials/extract",
        files={"file": ("resume.docx", buffer.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 200
    assert "项目经历" in response.json()["text"]
    assert "行动 | 访谈用户" in response.json()["text"]


def test_reject_unsupported_material() -> None:
    response = TestClient(app).post(
        "/api/v1/profile/materials/extract",
        files={"file": ("legacy.doc", b"not-a-word-file", "application/msword")},
    )

    assert response.status_code == 422
    assert "仅支持" in response.json()["detail"]
