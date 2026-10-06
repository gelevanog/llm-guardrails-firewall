"""Layer (b): a small open prompt-injection classifier on CPU (optional `classifier` extra).

The model is a sequence classifier fine-tuned to separate injections from benign prompts. It is loaded
lazily, runs with a fixed number of torch threads, and scores long texts in overlapping windows (the
highest window wins, so one injected paragraph in a long email is not averaged away).

Which model is the default was decided by measurement (README > Choosing the classifier); the
candidates are listed in `CANDIDATES` with their label conventions. All are ungated with permissive
licenses, so they download without accepting terms.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bulwark.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Candidate:
    model: str
    revision: str
    license: str
    params_m: int
    tokenizer: str | None = None
    """Tokenizer repo when the model repo does not ship one."""


CANDIDATES: dict[str, Candidate] = {
    "protectai/deberta-v3-base-prompt-injection-v2": Candidate(
        "protectai/deberta-v3-base-prompt-injection-v2", "90c9989b1a342275dd0d1a95aad283c04e075671", "Apache-2.0", 184
    ),
    "PreambleAI/prompt-injection-defense": Candidate(
        "PreambleAI/prompt-injection-defense",
        "cd7bfdb8d76278bbad4a3a0f637a4cba56ef4322",
        "Apache-2.0",
        150,
        tokenizer="answerdotai/ModernBERT-base",
    ),
    "madhurjindal/Jailbreak-Detector": Candidate(
        "madhurjindal/Jailbreak-Detector", "93883e180f4d6585a0afae703095ba4a9ebce888", "MIT", 66
    ),
}
DEFAULT_CLASSIFIER_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

_POSITIVE_LABELS = {"injection", "jailbreak", "label_1", "1", "untrusted", "unsafe", "malicious", "attack"}
_DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.spm"]


class ClassifierUnavailableError(RuntimeError):
    """The `classifier` extra is not installed or the model could not be loaded."""


def classifier_installed() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def fetch_model(
    model: str, *, revision: str | None = None, local_files_only: bool = False, tokenizer_only: bool = False
) -> str:
    """Local directory with the model: the path itself, or a Hugging Face snapshot (weights as safetensors)."""
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download

    candidate = CANDIDATES.get(model)
    return str(
        snapshot_download(
            model,
            revision=revision or (candidate.revision if candidate else None),
            allow_patterns=[p for p in _DOWNLOAD_PATTERNS if not (tokenizer_only and p == "*.safetensors")],
            local_files_only=local_files_only,
        )
    )


def model_cached(model: str) -> bool:
    """True when the model loads without network access (tests that need it are skipped otherwise)."""
    if not classifier_installed():
        return False
    try:
        fetch_model(model, local_files_only=True)
        candidate = CANDIDATES.get(model)
        if candidate and candidate.tokenizer:
            fetch_model(candidate.tokenizer, revision="main", local_files_only=True, tokenizer_only=True)
    except Exception:  # the hub raises several different errors for missing files
        return False
    return True


def windows(text: str, size: int = 1600, overlap: int = 200) -> list[str]:
    """Overlapping character windows; each is truncated to the model's token limit when encoded."""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text) - overlap, step)]


@dataclass
class InjectionClassifier:
    model_name: str = DEFAULT_CLASSIFIER_MODEL
    threads: int = 4
    max_length: int = 512
    batch_size: int = 8
    max_chars: int = 20_000
    name: str = "classifier"
    load_seconds: float | None = field(default=None, init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _positive: int = field(default=1, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not classifier_installed():
                raise ClassifierUnavailableError('the classifier needs: pip install "bulwark[classifier]"')
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            started = time.monotonic()
            if self.threads > 0:
                torch.set_num_threads(self.threads)
            candidate = CANDIDATES.get(self.model_name)
            try:
                path = fetch_model(self.model_name)
                tokenizer_path = (
                    fetch_model(candidate.tokenizer, revision="main", tokenizer_only=True)
                    if candidate and candidate.tokenizer
                    else path
                )
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
            except Exception as exc:  # network, missing files, incompatible checkpoint
                raise ClassifierUnavailableError(f"could not load classifier {self.model_name!r}: {exc}") from exc
            model.eval()
            labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            self._positive = next((i for i, label in labels.items() if label in _POSITIVE_LABELS), 1)
            self._model = model
            self.load_seconds = round(time.monotonic() - started, 2)
            log.info("classifier.loaded", model=self.model_name, seconds=self.load_seconds, labels=labels)

    def score_many(self, texts: list[str]) -> list[float]:
        """Probability of injection for each text (max over its windows)."""
        self.load()
        import torch

        pieces: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            for window in windows(text[: self.max_chars]):
                pieces.append((index, window if window.strip() else " "))
        scores = [0.0] * len(texts)
        with self._lock, torch.inference_mode():
            for start in range(0, len(pieces), self.batch_size):
                batch = pieces[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [text for _, text in batch],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)[:, self._positive].tolist()
                for (index, _), probability in zip(batch, probabilities, strict=True):
                    scores[index] = max(scores[index], float(probability))
        return [round(score, 4) for score in scores]

    def score(self, text: str) -> float:
        return self.score_many([text])[0]
