"""Evaluation datasets, downloaded at evaluation time into a local cache (nothing third-party is vendored).

* deepset/prompt-injections (Apache-2.0): 662 short prompts, English and German, labeled injection / benign.
* JailbreakBench (MIT): jailbreak prompts produced by PAIR, JBC (the "AIM" template), GCG and prompt + random
  search against gpt-4-0125-preview (positives), and the 100 benign behaviors of JBB-Behaviors (negatives).
* BIPIA (MIT for the email task and the attack texts): real-looking emails from the EmailQA test contexts,
  each used twice, clean and with one BIPIA text attack inserted at the start, middle or end.
* The hand-written set of this repository (see `handwritten.py`).

Every source is pinned to a revision (Hugging Face sha or Git commit) in configs/eval.yaml.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from bulwark.logging_config import get_logger

log = get_logger(__name__)

Context = Literal["input", "untrusted"]


class Sample(BaseModel):
    id: str
    dataset: str
    text: str
    label: int
    """1 = attack (injection or jailbreak), 0 = benign."""
    context: Context = "input"
    category: str = ""
    split: str = "test"
    lang: str = "en"


def _get(url: str, cache: Path) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    log.info("dataset.download", url=url)
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(response.content)
    return response.content


def _hf_revision(dataset: str) -> str:
    response = httpx.get(f"https://huggingface.co/api/datasets/{dataset}", timeout=30)
    response.raise_for_status()
    return str(response.json().get("sha", ""))


def load_deepset(cache_dir: Path, revision: str, splits: list[str]) -> list[Sample]:
    """Rows via the Hugging Face datasets-server (JSON, no parquet reader needed); checks the pinned revision."""
    current = _hf_revision("deepset/prompt-injections")
    if current and current != revision:
        log.warning("dataset.revision_changed", dataset="deepset/prompt-injections", pinned=revision, current=current)
    samples: list[Sample] = []
    for split in splits:
        offset = 0
        while True:
            url = (
                "https://datasets-server.huggingface.co/rows?dataset="
                f"{quote('deepset/prompt-injections', safe='')}&config=default&split={split}&offset={offset}&length=100"
            )
            data = json.loads(_get(url, cache_dir / "deepset" / revision / f"{split}-{offset}.json"))
            rows = data.get("rows", [])
            for row in rows:
                item = row["row"]
                samples.append(
                    Sample(
                        id=f"deepset-{split}-{row['row_idx']}",
                        dataset="deepset",
                        text=str(item["text"]),
                        label=int(item["label"]),
                        category="injection" if int(item["label"]) else "benign",
                        split=split,
                    )
                )
            offset += len(rows)
            if not rows or offset >= int(data.get("num_rows_total", 0)):
                break
    return samples


JBB_METHODS = {
    "PAIR": "PAIR/black_box",
    "JBC": "JBC/manual",
    "GCG": "GCG/transfer",
    "prompt_with_random_search": "prompt_with_random_search/black_box",
}


def load_jailbreakbench(
    cache_dir: Path, artifacts_commit: str, behaviors_revision: str, target: str, methods: list[str], per_method: int
) -> list[Sample]:
    samples: list[Sample] = []
    for method in methods:
        path = f"attack-artifacts/{JBB_METHODS[method]}/{target}.json"
        url = f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{artifacts_commit}/{path}"
        data = json.loads(_get(url, cache_dir / "jbb" / artifacts_commit / path))
        prompts = [j for j in data.get("jailbreaks", []) if j.get("prompt")]
        for item in prompts[:per_method]:
            samples.append(
                Sample(
                    id=f"jbb-{method}-{item['index']}",
                    dataset="jailbreakbench",
                    text=str(item["prompt"]),
                    label=1,
                    category=method,
                )
            )
    url = f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{behaviors_revision}/data/benign-behaviors.csv"
    raw = _get(url, cache_dir / "jbb" / behaviors_revision / "benign-behaviors.csv").decode("utf-8")
    for index, row in enumerate(csv.DictReader(io.StringIO(raw))):
        samples.append(
            Sample(
                id=f"jbb-benign-{index}",
                dataset="jailbreakbench",
                text=row["Goal"],
                label=0,
                category="benign behavior",
            )
        )
    return samples


def load_bipia(cache_dir: Path, commit: str, emails: int) -> list[Sample]:
    base = f"https://raw.githubusercontent.com/microsoft/BIPIA/{commit}/benchmark"
    contexts_raw = _get(f"{base}/email/test.jsonl", cache_dir / "bipia" / commit / "email-test.jsonl").decode("utf-8")
    attacks_raw = _get(f"{base}/text_attack_test.json", cache_dir / "bipia" / commit / "text_attack_test.json")
    contexts = [json.loads(line)["context"] for line in contexts_raw.splitlines() if line.strip()]
    unique = list(dict.fromkeys(contexts))[:emails]
    attacks = [(category, text) for category, texts in json.loads(attacks_raw).items() for text in texts]
    samples: list[Sample] = []
    for index, context in enumerate(unique):
        category, attack = attacks[index % len(attacks)]
        position = ("end", "start", "middle")[index % 3]
        if position == "end":
            attacked = f"{context}\n{attack}"
        elif position == "start":
            attacked = f"{attack}\n{context}"
        else:
            middle = len(context) // 2
            cut = context.rfind(" ", 0, middle) if context.rfind(" ", 0, middle) > 0 else middle
            attacked = f"{context[:cut]}\n{attack}\n{context[cut:]}"
        samples.append(
            Sample(
                id=f"bipia-{index}-clean",
                dataset="bipia",
                text=context,
                label=0,
                context="untrusted",
                category="clean email",
            )
        )
        samples.append(
            Sample(
                id=f"bipia-{index}-{position}",
                dataset="bipia",
                text=attacked,
                label=1,
                context="untrusted",
                category=f"{category} ({position})",
            )
        )
    return samples
