import pytest

from mcclub_rag.ingest.errors import OcrConfigError
from mcclub_rag.ingest.ocr import build_xberg_config, timeout_budget, verify_tessdata
from mcclub_rag.ingest.settings import IngestSettings


def _tessdir(tmp_path, langs):
    for lang in langs:
        (tmp_path / f"{lang}.traineddata").write_bytes(b"x")
    return IngestSettings(tessdata_prefix=tmp_path)


def test_verify_tessdata_ok(tmp_path):
    assert verify_tessdata(_tessdir(tmp_path, ["ara", "fra", "eng"])) == tmp_path


def test_verify_tessdata_missing_language(tmp_path):
    settings = _tessdir(tmp_path, ["fra", "eng"])
    with pytest.raises(OcrConfigError) as exc:
        verify_tessdata(settings)
    assert "ara" in exc.value.message
    assert "download_models.py" in exc.value.message


def test_verify_tessdata_empty_file_counts_as_missing(tmp_path):
    settings = _tessdir(tmp_path, ["ara", "fra", "eng"])
    (tmp_path / "eng.traineddata").write_bytes(b"")
    with pytest.raises(OcrConfigError, match="eng"):
        verify_tessdata(settings)


def test_verify_tessdata_missing_directory(tmp_path):
    with pytest.raises(OcrConfigError, match="ara, fra, eng"):
        verify_tessdata(IngestSettings(tessdata_prefix=tmp_path / "nope"))


def test_verify_real_tessdata(ocr_settings, tessdata_dir):
    assert verify_tessdata(ocr_settings) == tessdata_dir


@pytest.mark.parametrize(
    ("kind", "pages", "expected"),
    [
        ("pdf", 4, 60 + 10 * 4),
        ("pdf", None, 60),
        ("image", None, 70),
        ("docx", None, 60),
        ("xlsx", 3, 60),
        ("markdown", None, 60),
    ],
)
def test_timeout_budget(kind, pages, expected):
    assert timeout_budget(kind, pages, IngestSettings()) == expected


def test_xberg_config_matches_design(tmp_path):
    settings = IngestSettings(tessdata_prefix=tmp_path, max_pages=42, ocr_min_chars_per_page=50)
    config = build_xberg_config(settings, timeout_s=123)

    assert config.use_cache is False
    assert str(config.output_format) == "markdown"
    assert config.pages.extract_pages is True
    assert config.ocr_embedded_images is False
    assert config.extraction_timeout_secs == 123
    assert config.security_limits.max_pages == 42
    assert config.pdf_options.extract_images is False
    assert config.pdf_options.extract_tables is True

    ocr = config.ocr
    assert ocr.backend == "tesseract"
    assert ocr.language == ["ara", "fra", "eng"]
    assert ocr.tessdata_path == str(tmp_path)
    assert ocr.quality_thresholds.min_non_whitespace_per_page == 50

    tess = ocr.tesseract_config
    assert tess.language == ["ara", "fra", "eng"]
    assert tess.use_cache is False
    assert tess.enable_table_detection is False
    assert tess.output_format == "text"
    assert tess.psm == 3
