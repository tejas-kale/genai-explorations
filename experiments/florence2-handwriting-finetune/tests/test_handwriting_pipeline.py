import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "src" / f"{name}.py"
    module_name = name.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def image(path, size=(32, 20)):
    Image.new("RGB", size, "white").save(path)


def test_prepare_upload_finds_heic_images(tmp_path):
    prepare_upload = load_script("prepare_upload")
    (tmp_path / "page.HEIC").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x")

    assert [p.name for p in prepare_upload.images(tmp_path)] == ["page.HEIC"]


def test_sanitise_image_strips_metadata_and_bounds_size(tmp_path):
    prepare_upload = load_script("prepare_upload")
    src = tmp_path / "scan.jpg"
    dst = tmp_path / "out" / "scan.jpg"
    Image.new("RGB", (220, 120), "white").save(src, exif=Image.Exif())

    prepare_upload.sanitise_image(src, dst, max_side=64)

    out = Image.open(dst)
    assert max(out.size) == 64
    assert not out.getexif()


def test_remote_transcribe_retries_empty_output_and_writes_metadata(tmp_path, monkeypatch):
    remote_transcribe = load_script("gcp/remote_transcribe")
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "experiments"
    input_dir.mkdir()
    image(input_dir / "page one.png")

    class FakeBackend:
        def __init__(self):
            self.calls = 0

        def transcribe(self, path, prompt, max_new_tokens):
            self.calls += 1
            return "" if self.calls == 1 else "hello world"

    fake = FakeBackend()
    monkeypatch.setattr(remote_transcribe, "load_backend", lambda model, load_in_4bit: fake)

    remote_transcribe.main([
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--model", "Qwen/Qwen3.5-0.8B",
        "--max-new-tokens", "16",
        "--attempts", "3",
    ])

    slug = output_dir / "qwen3.5-0.8b"
    assert (slug / "page one.txt").read_text() == "hello world\n"
    data = json.loads((slug / "page one.json").read_text())
    assert data["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert data["attempt_count"] == 2
    assert data["status"] == "ok"
    assert data["runtime_seconds"] >= 0


def test_remote_transcribe_writes_parse_failed_after_empty_attempts(tmp_path, monkeypatch):
    remote_transcribe = load_script("gcp/remote_transcribe")
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "experiments"
    input_dir.mkdir()
    image(input_dir / "page.png")

    class FakeBackend:
        def transcribe(self, path, prompt, max_new_tokens):
            return "  "

    monkeypatch.setattr(remote_transcribe, "load_backend", lambda model, load_in_4bit: FakeBackend())

    remote_transcribe.main([
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--model", "google/gemma-4-E2B-it",
        "--max-new-tokens", "16",
        "--attempts", "2",
    ])

    data = json.loads((output_dir / "gemma-4-e2b" / "page.json").read_text())
    assert data["attempt_count"] == 2
    assert data["status"] == "parse-failed"


def test_apple_vision_ocr_finds_heic_images(tmp_path):
    apple_vision_ocr = load_script("apple_vision_ocr")
    (tmp_path / "page.HEIC").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x")

    assert [p.name for p in apple_vision_ocr.image_paths(tmp_path)] == ["page.HEIC"]


def test_apple_vision_ocr_writes_model_outputs(tmp_path, monkeypatch):
    apple_vision_ocr = load_script("apple_vision_ocr")
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "experiments"
    input_dir.mkdir()
    image(input_dir / "page.png")

    monkeypatch.setattr(apple_vision_ocr, "recognise_text", lambda path: ["hello", "world"])

    apple_vision_ocr.main([
        "--input", str(input_dir),
        "--output", str(output_dir),
    ])

    slug = output_dir / "apple-vision-ocr"
    assert (slug / "page.txt").read_text() == "hello\nworld\n"
    data = json.loads((slug / "page.json").read_text())
    assert data["model_id"] == "apple/Vision-VNRecognizeTextRequest"
    assert data["attempt_count"] == 1
    assert data["status"] == "ok"


def test_label_ui_prefers_existing_label_then_largest_model_draft(tmp_path):
    label_ui = load_script("label_ui")
    images = tmp_path / "data"
    experiments = tmp_path / "experiments"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image(images / "one.png")
    image(images / "two.png")
    (experiments / "qwen3.5-9b").mkdir(parents=True)
    (experiments / "qwen3.5-9b" / "one.txt").write_text("draft one\n")
    (experiments / "qwen3.5-9b" / "two.txt").write_text("draft two\n")
    (labels / "one.txt").write_text("edited one\n")

    state = label_ui.State(images, experiments, labels, "qwen3.5-9b")

    assert state.items()[0].stem == "one"
    assert state.text_for("one") == "edited one\n"
    assert state.text_for("two") == "draft two\n"


def test_label_ui_converts_heic_with_sips_when_pillow_cannot_open(tmp_path, monkeypatch):
    label_ui = load_script("label_ui")
    src = tmp_path / "page.HEIC"
    src.write_bytes(b"heic")
    real_open = label_ui.Image.open

    def fake_open(path):
        if Path(path).suffix.lower() == ".heic":
            raise UnidentifiedImageError("no")
        return real_open(path)

    def fake_run(cmd, check):
        Image.new("RGB", (8, 8), "white").save(cmd[-1])

    monkeypatch.setattr(label_ui.Image, "open", fake_open)
    monkeypatch.setattr(label_ui.subprocess, "run", fake_run)

    assert label_ui.jpeg_bytes(src).startswith(b"\xff\xd8")


def test_label_ui_saves_label(tmp_path):
    label_ui = load_script("label_ui")
    images = tmp_path / "data"
    experiments = tmp_path / "experiments"
    labels = tmp_path / "labels"
    images.mkdir()
    image(images / "one.png")

    state = label_ui.State(images, experiments, labels, "qwen3.5-9b")
    state.save("one", "ground truth\n")

    assert (labels / "one.txt").read_text() == "ground truth\n"


def test_evaluate_prints_label_count_and_model_means(tmp_path, capsys):
    evaluate = load_script("evaluate_handwriting")
    experiments = tmp_path / "experiments"
    labels = tmp_path / "labels"
    (experiments / "model-a").mkdir(parents=True)
    (experiments / "model-b").mkdir(parents=True)
    labels.mkdir()
    (experiments / "model-a" / "one.txt").write_text("Hello, world!\n")
    (experiments / "model-b" / "one.txt").write_text("hello word\n")
    (experiments / "model-a" / "two.txt").write_text("ignored\n")
    (labels / "one.txt").write_text("hello world\n")

    evaluate.main([
        "--experiments", str(experiments),
        "--labels", str(labels),
    ])

    out = capsys.readouterr().out
    assert "labels: 1" in out
    assert "words: 2" in out
    assert "characters: 11" in out
    assert "model-a" in out
    assert "mean_normalised_wer" in out
    assert "0.0000" in out
