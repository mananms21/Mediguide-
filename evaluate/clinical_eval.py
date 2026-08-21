"""
MEDIGUIDE — Clinical Evaluation Module
=======================================
Four-level evaluation framework addressing the limitation that generic
BERTScore cannot distinguish clinically different but semantically
similar terms (e.g. "heart" vs "lung", "left" vs "right ventricle").

Levels:
  1. ClinicalBERTScorer   — BiomedBERT semantic similarity
  2. MedicalEntityScorer  — scispacy NER entity precision/recall/F1
  3. NLIConsistencyScorer — roberta-large-mnli contradiction detection
  4. HallucinationScorer  — entity grounding / specificity

Usage:
    from evaluate.clinical_eval import run_all_clinical_metrics
    results = run_all_clinical_metrics(preds, refs, questions)
"""

from __future__ import annotations
import warnings
from typing import Optional

warnings.filterwarnings("ignore")


# ── Level 1 — Clinical BERTScore (BiomedBERT) ─────────────────────────

class ClinicalBERTScorer:
    """
    BERTScore computed with a biomedical-domain model instead of the
    generic bert-base-uncased / roberta-large default.

    Model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    Trained on 29M PubMed abstracts + PubMed Central full text.
    In this embedding space, "heart" and "lung" are further apart than
    in general BERT, making semantic similarity clinically meaningful.
    """

    MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    def score(self, preds, refs, device="cpu", verbose=True):
        try:
            from bert_score import score as bs
            P, R, F1 = bs(
                preds, refs,
                model_type=self.MODEL,
                lang="en",
                verbose=verbose,
                device=device,
                rescale_with_baseline=False,
            )
            return {
                "clinical_bertscore_p":  round(float(P.mean()), 4),
                "clinical_bertscore_r":  round(float(R.mean()), 4),
                "clinical_bertscore_f1": round(float(F1.mean()), 4),
            }
        except Exception as e:
            print(f"[ClinicalBERTScorer] Error: {e}")
            return {"clinical_bertscore_p": None, "clinical_bertscore_r": None,
                    "clinical_bertscore_f1": None}


# ── Level 2 — Medical NER Entity F1 ───────────────────────────────────

class MedicalEntityScorer:
    """
    Extracts medical/scientific named entities with scispacy and computes
    precision, recall, F1 between prediction and reference entity sets.
    Catches wrong anatomy, wrong disease names, missing key entities.
    """

    _NLP = None

    @classmethod
    def _load_nlp(cls):
        if cls._NLP is None:
            try:
                import spacy
                cls._NLP = spacy.load("en_core_sci_md")
            except OSError:
                try:
                    import spacy
                    cls._NLP = spacy.load("en_core_sci_sm")
                except OSError:
                    cls._NLP = False
        return cls._NLP

    def _extract(self, text):
        nlp = self._load_nlp()
        if not nlp:
            import re
            tokens = re.findall(r"\b[A-Za-z][a-z]{3,}\b", text)
            return {t.lower() for t in tokens}
        doc = nlp(text[:5000])
        return {ent.text.lower().strip() for ent in doc.ents if len(ent.text) > 1}

    def score_pair(self, pred, ref):
        pred_ents = self._extract(pred)
        ref_ents  = self._extract(ref)
        if not ref_ents and not pred_ents:
            return 1.0, 1.0, 1.0
        if not pred_ents:
            return 0.0, 0.0, 0.0
        if not ref_ents:
            return 1.0, 0.0, 0.0
        tp = len(pred_ents & ref_ents)
        p  = tp / len(pred_ents)
        r  = tp / len(ref_ents)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    def score(self, preds, refs):
        import numpy as np
        ps, rs, f1s = [], [], []
        for pred, ref in zip(preds, refs):
            p, r, f1 = self.score_pair(pred, ref)
            ps.append(p); rs.append(r); f1s.append(f1)
        return {
            "entity_precision": round(float(np.mean(ps)), 4),
            "entity_recall":    round(float(np.mean(rs)), 4),
            "entity_f1":        round(float(np.mean(f1s)), 4),
        }


# ── Level 3 — NLI Factual Consistency ────────────────────────────────

class NLIConsistencyScorer:
    """
    Uses Natural Language Inference to detect clinical contradictions.
    Model: roberta-large-mnli
    premise = reference answer (ground truth)
    hypothesis = model prediction

    Labels: ENTAILMENT (consistent), NEUTRAL, CONTRADICTION (dangerous)
    The contradiction_rate is the key clinical safety metric.
    """

    MODEL = "roberta-large-mnli"
    _TOK  = None
    _MDL  = None

    @classmethod
    def _load(cls, device="cpu"):
        if cls._TOK is None:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            cls._TOK = AutoTokenizer.from_pretrained(cls.MODEL)
            cls._MDL = AutoModelForSequenceClassification.from_pretrained(cls.MODEL)
        import torch
        cls._MDL = cls._MDL.to(device).eval()
        return cls._TOK, cls._MDL

    def predict_pair(self, premise, hypothesis, device="cpu"):
        import torch
        tok, mdl = self._load(device)
        enc = tok(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # roberta-large-mnli: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
        return {
            "contradiction": round(probs[0].item(), 4),
            "neutral":       round(probs[1].item(), 4),
            "entailment":    round(probs[2].item(), 4),
        }

    def score(self, preds, refs, device="cpu"):
        import numpy as np
        cs, ns, es = [], [], []
        for pred, ref in zip(preds, refs):
            r = self.predict_pair(ref[:1024], pred[:1024], device=device)
            cs.append(r["contradiction"])
            ns.append(r["neutral"])
            es.append(r["entailment"])
        return {
            "contradiction_rate": round(float(np.mean(cs)), 4),
            "neutral_rate":       round(float(np.mean(ns)), 4),
            "entailment_rate":    round(float(np.mean(es)), 4),
        }


# ── Level 4 — Hallucination Score ────────────────────────────────────

class HallucinationScorer:
    """
    Measures what fraction of medical entities in the prediction are NOT
    grounded in the question or reference. Ungrounded entities are
    potential hallucinations.
    hallucination_rate = 1 - (grounded_entities / total_pred_entities)
    """

    _ner = None

    @property
    def ner(self):
        if self._ner is None:
            self._ner = MedicalEntityScorer()
        return self._ner

    def score_triplet(self, question, pred, ref):
        pred_ents  = self.ner._extract(pred)
        known_ents = self.ner._extract(question) | self.ner._extract(ref)
        if not pred_ents:
            return 0.0
        grounded = len(pred_ents & known_ents)
        return round(1.0 - grounded / len(pred_ents), 4)

    def score(self, questions, preds, refs):
        import numpy as np
        rates = [self.score_triplet(q, p, r) for q, p, r in zip(questions, preds, refs)]
        return {"hallucination_rate": round(float(np.mean(rates)), 4)}


# ── Convenience wrapper ────────────────────────────────────────────────

def run_all_clinical_metrics(preds, refs, questions, device="cpu", verbose=True):
    """Run all four clinical evaluation levels, return merged dict."""
    results = {}
    print("  [1/4] Clinical BERTScore (BiomedBERT)…")
    results.update(ClinicalBERTScorer().score(preds, refs, device=device, verbose=verbose))
    print("  [2/4] Medical NER Entity F1 (scispacy)…")
    results.update(MedicalEntityScorer().score(preds, refs))
    print("  [3/4] NLI Factual Consistency (roberta-large-mnli)…")
    results.update(NLIConsistencyScorer().score(preds, refs, device=device))
    print("  [4/4] Hallucination Rate…")
    results.update(HallucinationScorer().score(questions, preds, refs))
    return results
