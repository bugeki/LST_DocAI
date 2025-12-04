"""
Inference helpers using real Hugging Face models.

Notes
-----
- First call will download models from Hugging Face; this can take a while.
- Models are kept in global pipelines so subsequent requests are fast.
- Chosen models:
  - NER: a Turkish BERT NER model
  - LLM: a multilingual mT5 model used for Turkish summarization
"""

from functools import lru_cache
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import torch


NER_MODEL_NAME = "savasy/bert-base-turkish-ner-cased"
SUM_MODEL_NAME = "google/mt5-small"


def _device() -> int:
    """Return GPU device id if available, otherwise CPU (-1)."""
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def get_ner_pipeline():
    """Lazy-load and cache the NER pipeline."""
    tok = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    return pipeline("ner", model=model, tokenizer=tok, grouped_entities=True, device=_device())


@lru_cache(maxsize=1)
def get_summarizer_pipeline():
    """Lazy-load and cache the summarization (LLM) pipeline."""
    tok = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)
    return pipeline("text2text-generation", model=model, tokenizer=tok, device=_device())


def ner_infer(text: str) -> List[Dict]:
    """
    Run NER over the input text and return a list of entities.

    Each entity dict contains:
      - text: the surface form
      - label: the predicted entity type
      - score: confidence
      - start / end: character offsets (best-effort from token spans)
    """
    if not text.strip():
        return []

    nlp_ner = get_ner_pipeline()
    raw_entities = nlp_ner(text)

    entities: List[Dict] = []
    for ent in raw_entities:
        entities.append(
            {
                "text": ent.get("word") or ent.get("entity_group"),
                "label": ent.get("entity_group") or ent.get("entity"),
                "score": float(ent.get("score", 0.0)),
                "start": ent.get("start"),
                "end": ent.get("end"),
            }
        )
    return entities


def llm_extract(text: str) -> str:
    """
    Use a seq2seq LLM to summarize / extract key info in Turkish.

    For now we prompt the model to produce a short Turkish summary.
    You can later swap this to a task-specific, instruction-tuned model.
    """
    if not text.strip():
        return ""

    summarizer = get_summarizer_pipeline()

    prompt = (
        "Aşağıdaki metni kısa ve öz bir şekilde özetle, metindeki kişi, kurum, tarih "
        "ve talep/şikayet bilgisini vurgula:\n\n"
        f"{text}"
    )

    out = summarizer(
        prompt,
        max_length=128,
        num_beams=4,
        do_sample=False,
    )

    return out[0]["generated_text"].strip() if out and "generated_text" in out[0] else ""

