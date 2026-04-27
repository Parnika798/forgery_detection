#Latest API.PY
"""
Receipt Forgery Detection — FastAPI Backend (v4.1)
==================================================
Bug-fixes over v4 (the 5 bugs found from the test receipt):

  BUG 1 (PRIMARY): extract_structured_fields() was line-by-line only.
    Thermal printers emit "TOTAL." on one line, "5,800.00" on the next.
    The amount fell into items[], total stayed None, validate_totals() returned 0.0.
    Fix: _join_split_lines() pre-processor + inline look-ahead in extractor.

  BUG 2: _ocr() total regex required keyword+amount on ONE line.
    Fix: two-line fallback using _join_split_lines() result.

  BUG 3: validate_totals() returned early when total is None.
    Fix: _infer_total_from_items() — when max(items) >> rest, treat as total.

  BUG 4: _text() tax-math and net-payable regexes had the same single-line
    assumption. Fix: run all regexes on _join_split_lines(text).

  BUG 5: sticker amount contaminated items[]. After Bug 1 fix total is
    correctly extracted so this is mostly moot, but _infer_total_from_items()
    now cleanly separates the inferred total from the item pool.
"""

from __future__ import annotations
import os, re, json, pickle, warnings, base64
import numpy as np
import cv2

warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Receipt Scanner API v4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

FEATURE_COLS = [
    "max_amount", "n_amounts", "n_dates", "valid_date_fmt",
    "has_currency", "line_count", "total_to_max_ratio",
    "amount_std", "unique_amount_ratio", "has_total_keyword",
    "avg_word_conf", "text_density", "round_number_ratio",
]

CRITICAL_KEYWORDS = [
    "total", "subtotal", "grand", "amount", "payable",
    "net", "due", "balance", "paid", "tax", "gst", "vat",
]

# ── keyword patterns ──────────────────────────────────────────────────────────
_KW_TOTAL    = re.compile(
    r"\b(gr(?:and)?\s*total|total\s*amt?|total\s*amount|amount\s*due"
    r"|net\s*payable|bill\s*total|final\s*total|payable\s*amount"
    r"|balance\s*due|total\s*bill|total)\b", re.I)
_KW_SUBTOTAL = re.compile(
    r"\b(sub\s*total|subtotal|taxable\s*(?:amount|amt)|base\s*amount)\b", re.I)
_KW_TAX      = re.compile(
    r"\b(s\.?g\.?s\.?t|c\.?g\.?s\.?t|i\.?g\.?s\.?t|gst|vat|tax|cess|surcharge"
    r"|service\s*charge|service\s*tax)\b", re.I)
_KW_DISCOUNT = re.compile(
    r"\b(discount|offer|promo|coupon|savings?|less)\b", re.I)
_KW_SKIP     = re.compile(
    r"\b(qty|quantity|rate|price\s*per|unit|mrp|hsn|sac|item|product|description"
    r"|ph(?:one)?|tel|mobile|date|bill\s*no|invoice|order|table|cover|gstin|cin"
    r"|pan|upi|card|cash|ref|receipt|no\.?|#)\b", re.I)
_AMOUNT_PAT  = re.compile(r"(?:rs\.?\s*|RS\s*|₹\s*)?([\d,]+\.?\d*)", re.I)
# Line that contains ONLY an amount (with optional currency symbol)
_AMOUNT_ONLY = re.compile(r"^\s*(?:rs\.?\s*|₹\s*)?[\d,]+\.?\d*\s*$", re.I)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lev(a: str, b: str) -> int:
    if len(a) < len(b): return _lev(b, a)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(0 if ca==cb else 1)))
        prev = curr
    return prev[-1]


def _fuzzy(text: str, keyword: str, max_dist: int = 2) -> bool:
    norm = (text.lower().replace("0","o").replace("1","i")
            .replace("3","e").replace("@","a"))
    kw = keyword.lower()
    if kw in norm: return True
    for tok in re.findall(r"[a-z]{2,}", norm):
        if abs(len(tok)-len(kw)) <= 2 and _lev(tok, kw) <= max_dist:
            return True
    return False


def _parse_amount(s: str) -> float | None:
    """Parse a string into a receipt-range float; None if out of range."""
    try:
        v = float(re.sub(r"[,\s]", "", s).strip())
        return v if 0.5 <= v <= 1_000_000 else None
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 1 + FIX 4 — _join_split_lines()
#
#  Thermal printer OCR often gives:
#      "TOTAL."        <- keyword only, no amount
#      "5,800.00"      <- amount only, no keyword
#
#  This pre-processor detects those pairs and merges them:
#      "TOTAL.  5,800.00"
#
#  All downstream regex checks (in _text(), _ocr(), validate_critical_fields)
#  run on this joined text so they see keyword + value on one line.
#
#  Conservative: only joins when the next non-empty line is PURELY an amount.
# ─────────────────────────────────────────────────────────────────────────────

def _join_split_lines(text: str) -> str:
    lines  = text.splitlines()
    out    = []
    used   = set()      # indices already merged into a prior line

    for i, line in enumerate(lines):
        if i in used:
            out.append("")
            continue

        has_kw = bool(
            _KW_TOTAL.search(line) or _KW_SUBTOTAL.search(line) or
            _KW_TAX.search(line)   or _KW_DISCOUNT.search(line)
        )
        has_amt = bool(re.search(r"[\d,]+\.?\d*", line))

        if has_kw and not has_amt:
            # Search forward up to 2 lines for a pure-amount line
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                if _AMOUNT_ONLY.match(nxt):
                    out.append(line.rstrip() + "  " + nxt)
                    used.add(j)
                    break
            else:
                out.append(line)
        else:
            out.append(line)

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 3 — _infer_total_from_items()
#
#  Last-resort: when no total keyword was found (or keyword/amount split
#  wasn't bridged), the largest amount may still be the total if it is
#  clearly dominant (>= 1.5x second-largest AND >= sum of all others).
#
#  Returns (inferred_total, items_without_inferred_total) or (None, items).
# ─────────────────────────────────────────────────────────────────────────────

def _infer_total_from_items(items: list) -> tuple:
    if len(items) < 2:
        return None, items
    sd   = sorted(items, reverse=True)
    cand = sd[0]
    rest = sd[1:]
    if cand >= sd[1] * 1.5 and cand >= sum(rest) * 0.90:
        return cand, rest
    return None, items


# ─────────────────────────────────────────────────────────────────────────────
#  NEW — extract_structured_fields()  (v4.1: look-ahead + join)
# ─────────────────────────────────────────────────────────────────────────────

def extract_structured_fields(text: str) -> dict:
    """
    Classify each receipt line into: total/subtotal/tax/discount/item/skip.

    v4.1 changes:
      • Runs on _join_split_lines(text) to handle split keyword/amount lines.
      • Inline look-ahead for any remaining keyword-only lines.
      • _infer_total_from_items() fallback when total keyword extraction fails.
    """
    joined = _join_split_lines(text)
    result = {
        "items": [], "taxes": [], "discounts": [],
        "subtotal": None, "total": None, "extra_totals": [],
        "raw_fields": [], "_inferred_total": False,
    }

    lines = [l.strip() for l in joined.splitlines()]
    # Work with a mutable list so look-ahead can nullify consumed lines
    lines_mut = list(lines)

    for idx, line in enumerate(lines_mut):
        if not line:
            continue

        amounts_on_line = []
        for m in _AMOUNT_PAT.finditer(line):
            v = _parse_amount(m.group(1))
            if v is not None:
                amounts_on_line.append(v)

        # If keyword present but no amount, look ahead one more time
        # (belt-and-suspenders: _join_split_lines should have handled this)
        if not amounts_on_line:
            is_kw = bool(
                _KW_TOTAL.search(line) or _KW_SUBTOTAL.search(line) or
                _KW_TAX.search(line)
            )
            if is_kw:
                for j in range(idx + 1, min(idx + 3, len(lines_mut))):
                    nxt = lines_mut[j].strip()
                    if not nxt:
                        continue
                    if _AMOUNT_ONLY.match(nxt):
                        v = _parse_amount(re.sub(r"[₹RS\s]", "", nxt))
                        if v:
                            amounts_on_line = [v]
                            lines_mut[j] = ""   # consume
                        break
            if not amounts_on_line:
                continue

        rightmost = amounts_on_line[-1]

        if _KW_TOTAL.search(line):
            if result["total"] is None:
                result["total"] = rightmost
            else:
                result["extra_totals"].append(rightmost)
            result["raw_fields"].append({"label": "total", "line": line, "value": rightmost})

        elif _KW_SUBTOTAL.search(line):
            result["subtotal"] = rightmost
            result["raw_fields"].append({"label": "subtotal", "line": line, "value": rightmost})

        elif _KW_TAX.search(line):
            result["taxes"].append(rightmost)
            result["raw_fields"].append({"label": "tax", "line": line, "value": rightmost})

        elif _KW_DISCOUNT.search(line):
            result["discounts"].append(rightmost)
            result["raw_fields"].append({"label": "discount", "line": line, "value": rightmost})

        elif _KW_SKIP.search(line):
            result["raw_fields"].append({"label": "skip", "line": line, "value": rightmost})

        else:
            result["items"].append(rightmost)
            result["raw_fields"].append({"label": "item", "line": line, "value": rightmost})

    # FIX 3: infer total from item distribution if extraction still failed
    if result["total"] is None and result["items"]:
        inferred, remaining = _infer_total_from_items(result["items"])
        if inferred is not None:
            result["total"]           = inferred
            result["items"]           = remaining
            result["_inferred_total"] = True

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  validate_totals()
# ─────────────────────────────────────────────────────────────────────────────

def validate_totals(structured: dict, tolerance_pct: float = 0.02,
                    min_tolerance: float = 2.0) -> tuple:
    """
    Try multiple arithmetic strategies to check sum(items) vs stated total.
    Returns (score, flags).  score=0.55 → definitive mismatch.
    """
    flags = []
    total = structured.get("total")
    if total is None or total <= 0:
        return 0.0, []

    items     = structured.get("items", [])
    taxes     = structured.get("taxes", [])
    discounts = structured.get("discounts", [])
    subtotal  = structured.get("subtotal")
    inferred  = structured.get("_inferred_total", False)

    item_sum  = round(sum(items), 2)
    tax_sum   = round(sum(taxes), 2)
    disc_sum  = round(sum(discounts), 2)
    tolerance = max(total * tolerance_pct, min_tolerance)

    def _ok(v): return abs(v - total) <= tolerance

    if len(items) <= 1 and subtotal is None:
        return 0.0, []

    tried, passed, mismatches = 0, 0, []

    if items and (taxes or discounts):
        tried += 1
        c = round(item_sum + tax_sum - disc_sum, 2)
        if _ok(c): passed += 1
        else: mismatches.append(f"items({item_sum:.2f})+tax({tax_sum:.2f})-disc({disc_sum:.2f})={c:.2f}≠{total:.2f}")

    if subtotal is not None and (taxes or discounts):
        tried += 1
        c = round(subtotal + tax_sum - disc_sum, 2)
        if _ok(c): passed += 1
        else: mismatches.append(f"subtotal({subtotal:.2f})+tax({tax_sum:.2f})-disc({disc_sum:.2f})={c:.2f}≠{total:.2f}")

    if items and taxes:
        tried += 1
        c = round(item_sum + tax_sum, 2)
        if _ok(c): passed += 1
        else: mismatches.append(f"items({item_sum:.2f})+tax({tax_sum:.2f})={c:.2f}≠{total:.2f}")

    if len(items) >= 2:
        tried += 1
        if _ok(item_sum): passed += 1
        else: mismatches.append(f"sum_items({item_sum:.2f})≠total({total:.2f})")

    if subtotal is not None and not taxes and not items:
        tried += 1
        if _ok(subtotal): passed += 1
        else: mismatches.append(f"subtotal({subtotal:.2f})≠total({total:.2f})")

    if tried == 0:
        return 0.0, []

    tag = " [total inferred from amounts]" if inferred else ""

    if passed == 0:
        flags.append(f"TOTAL MISMATCH (all {tried} checks failed){tag}: {mismatches[0]}")
        return 0.55, flags

    if passed < tried and tried >= 2:
        flags.append(f"Partial mismatch ({passed}/{tried} passed){tag}: " +
                     "; ".join(mismatches[:2]))
        return 0.30, flags

    return 0.0, []


# ─────────────────────────────────────────────────────────────────────────────
#  validate_critical_fields()
# ─────────────────────────────────────────────────────────────────────────────

def validate_critical_fields(text: str, structured: dict) -> tuple:
    flags = []
    score = 0.0
    total    = structured.get("total")
    subtotal = structured.get("subtotal")
    extras   = structured.get("extra_totals", [])
    taxes    = structured.get("taxes", [])

    all_totals = [t for t in ([total] + extras) if t is not None]
    if len(all_totals) >= 2:
        spread = max(all_totals) - min(all_totals)
        if spread > max(max(all_totals) * 0.02, 2.0):
            flags.append(f"Multiple total fields disagree: {all_totals} (spread={spread:.2f})")
            score += 0.35

    if subtotal is not None and total is not None:
        if subtotal > total * 1.02:
            flags.append(f"Subtotal ({subtotal:.2f}) exceeds Total ({total:.2f}) — impossible")
            score += 0.45

    if total is not None:
        for t in taxes:
            if t > total:
                flags.append(f"Tax ({t:.2f}) exceeds total ({total:.2f}) — impossible")
                score += 0.35
                break

    # Digit-prepend/append — use joined text for reliability (FIX 4)
    joined = _join_split_lines(text)
    total_m = re.search(
        r"(?:total\s*(?:amt?|amount)?|amount\s*due|net\s*payable)"
        r"[^:\d]*(?:rs\.?\s*)?([\d,]+\.?\d*)", joined, re.I)
    sub_m   = re.search(
        r"sub\s*total[^:\d]*(?:rs\.?\s*)?([\d,]+\.?\d*)", joined, re.I)
    if total_m and sub_m:
        ts = re.sub(r"[,.\s]","", total_m.group(1).split(".")[0])
        ss = re.sub(r"[,.\s]","", sub_m.group(1).split(".")[0])
        if len(ts) > len(ss) and ts.endswith(ss):
            flags.append(f"Digit prepended to subtotal→total: '{ss}'→'{ts}'")
            score += 0.70
        elif len(ts) > len(ss) and ts.startswith(ss):
            flags.append(f"Digit appended to subtotal→total: '{ss}'→'{ts}'")
            score += 0.70
        elif len(ss) > len(ts) and ss.endswith(ts):
            flags.append(f"Digit prepended to total: '{ts}'→'{ss}'")
            score += 0.70

    return float(min(score, 1.0)), flags


# ─────────────────────────────────────────────────────────────────────────────
#  _logical_anomaly_score() — entry point for semantic validation
# ─────────────────────────────────────────────────────────────────────────────

def _logical_anomaly_score(text: str, ocr_fields: dict) -> tuple:
    if not text or len(text.strip()) < 10:
        return 0.0, []
    structured = extract_structured_fields(text)
    ts, tf     = validate_totals(structured)
    cs, cf     = validate_critical_fields(text, structured)
    ocr_fields["_structured"] = structured
    return float(min(ts + cs, 1.0)), tf + cf


# ─────────────────────────────────────────────────────────────────────────────
#  AnomalyDetector — caches logical score from _text()
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import __main__


class AnomalyDetector:
    def __init__(self, max_amount: float = 10_000.0, contamination: float = 0.15):
        self.max_amount    = max_amount
        self.contamination = contamination
        self.scaler        = StandardScaler()
        self.iso_forest    = IsolationForest(
            contamination=contamination, n_estimators=300,
            max_samples="auto", random_state=42)
        self.fitted = False

    def fit(self, df_feats) -> None:
        real = df_feats[df_feats["label"] == 0]
        if len(real) < 10: real = df_feats
        X = real[FEATURE_COLS].values
        self.scaler.fit(X)
        self.iso_forest.fit(self.scaler.transform(X))
        self.fitted = True

    def score(self, ocr_fields: dict) -> tuple:
        flags, score = [], 0.0
        max_amt  = ocr_fields.get("max_amount", 0)
        avg_conf = ocr_fields.get("avg_word_conf", 80.0)

        if max_amt > self.max_amount:
            flags.append(f"Amount {max_amt:,.2f} exceeds expected threshold")
            score += 0.30
        if max_amt > self.max_amount * 3: score += 0.10
        if avg_conf < 50.0:
            flags.append(f"Low OCR confidence ({avg_conf:.0f}%) — font inconsistency")
            score += 0.20
        elif avg_conf < 70.0: score += 0.08
        if not _fuzzy(ocr_fields.get("raw_text", ""), "total", 2):
            flags.append("TOTAL keyword missing or corrupted")
            score += 0.15
        if ocr_fields.get("amount_std", 0.0) > 500:
            flags.append(f"High amount variance (std={ocr_fields['amount_std']:.0f})")
            score += 0.10
        if not ocr_fields.get("valid_date_fmt", 1):
            flags.append("Date format mismatch"); score += 0.15
        if not ocr_fields.get("has_currency", 1):
            flags.append("No currency symbol detected"); score += 0.08
        if ocr_fields.get("total_to_max_ratio", 1.0) > 2.0:
            flags.append(f"Total/max ratio suspicious ({ocr_fields['total_to_max_ratio']:.2f})")
            score += 0.12
        if ocr_fields.get("n_amounts", 1) == 0:
            flags.append("No monetary amounts found"); score += 0.12
        if self.fitted:
            feats   = np.array([ocr_fields.get(c, 0.0) for c in FEATURE_COLS]).reshape(1,-1)
            iso_raw = self.iso_forest.decision_function(self.scaler.transform(feats))[0]
            iso_n   = float(np.clip(-iso_raw + 0.3, 0.0, 1.0))
            score  += iso_n * 0.25
            if iso_n > 0.45: flags.append(f"Statistical outlier (IF score {iso_n:.2f})")

        raw_text = ocr_fields.get("raw_text", "")
        if "_logical_score" in ocr_fields:
            log_score = ocr_fields["_logical_score"]
            log_flags = ocr_fields.get("_logical_flags", [])
        else:
            log_score, log_flags = _logical_anomaly_score(raw_text, ocr_fields)
            ocr_fields["_logical_score"] = log_score
            ocr_fields["_logical_flags"] = log_flags

        print("===== DEBUG LOGIC =====")
        print("LOG SCORE:", log_score)
        print("LOG FLAGS:", log_flags)
        print("STRUCTURED:", ocr_fields.get("_structured"))    

        if log_score > 0:
    # 🔥 Increase impact of logical reasoning
            score += log_score * 0.80

    # 🔥 Strong mismatch should strongly influence anomaly
            if log_score >= 0.5:
                flags.append("Strong logical inconsistency detected")
                score += 0.3  # additional boost for clear fraud signals

            flags.extend(log_flags)

        return float(min(score, 1.0)), flags


__main__.AnomalyDetector = AnomalyDetector

_models: dict = {}


def _load_models() -> None:
    import torch, torch.nn as nn, timm
    import segmentation_models_pytorch as smp

    base  = os.path.dirname(__file__)
    paths = {
        "clf" : os.path.join(base, "classifier_final.pth"),
        "seg" : os.path.join(base, "unet_final.pth"),
        "anom": os.path.join(base, "anomaly_model.pkl"),
        "cfg" : os.path.join(base, "config.json"),
    }
    missing = [k for k, p in paths.items() if k != "cfg" and not os.path.exists(p)]
    if missing: raise RuntimeError(f"Missing model files: {missing}")

    cfg = {}
    if os.path.exists(paths["cfg"]):
        with open(paths["cfg"]) as f: cfg = json.load(f)

    img_size    = int(cfg.get("img_size", 320))
    backbone    = cfg.get("backbone", "efficientnet_b3")
    seg_encoder = cfg.get("seg_encoder", "efficientnet-b3")
    threshold   = float(cfg.get("optimal_threshold", cfg.get("threshold", 0.38)))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Clf(nn.Module):
        def __init__(self, bb, v2=True):
            super().__init__()
            self.enc = timm.create_model(bb, pretrained=False, num_classes=0, global_pool="avg")
            feat = self.enc.num_features
            if v2:
                self.head = nn.Sequential(
                    nn.BatchNorm1d(feat), nn.Dropout(0.4), nn.Linear(feat, 512),
                    nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, 2))
            else:
                self.head = nn.Sequential(
                    nn.Dropout(0.4), nn.Linear(feat, 256),
                    nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        def forward(self, x): return self.head(self.enc(x))

    ck    = torch.load(paths["clf"], map_location=device, weights_only=False)
    state = ck.get("state", ck)
    clf   = Clf(backbone, v2=True).to(device)
    try:
        clf.load_state_dict(state, strict=True)
    except RuntimeError:
        clf = Clf(backbone, v2=False).to(device)
        clf.load_state_dict(state, strict=True)
    clf.eval()

    unet = smp.Unet(encoder_name=seg_encoder, encoder_weights=None,
                    in_channels=3, classes=1, activation="sigmoid").to(device)
    unet.load_state_dict(torch.load(paths["seg"], map_location=device, weights_only=True))
    unet.eval()

    with open(paths["anom"], "rb") as f: anom = pickle.load(f)

    _models.update({"clf": clf, "unet": unet, "anom": anom, "device": device,
                    "img_size": img_size, "threshold": threshold})
    print(f"Models loaded on {device}  threshold={threshold}")


@app.on_event("startup")
def startup() -> None:
    try: _load_models()
    except Exception as e: print(f"Model load failed: {e}")


def _preprocess(img_rgb, size):
    import torch
    r = cv2.resize(img_rgb, (size, size))
    n = (r.astype(np.float32) / 255.0 - np.array(MEAN)) / np.array(STD)
    return torch.tensor(n.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 2 — _ocr(): two-line total extraction fallback
# ─────────────────────────────────────────────────────────────────────────────

def _ocr(img_rgb: np.ndarray) -> dict:
    try:
        import pytesseract
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray  = clahe.apply(gray)
        gray  = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
        th    = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 15, 8)

        text = pytesseract.image_to_string(gray, config="--psm 6 --oem 3")
        if len(text.strip()) < 20:
            text = pytesseract.image_to_string(th, config="--psm 4 --oem 3")

        word_boxes = []
        avg_conf   = 50.0
        try:
            d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                          config="--psm 6 --oem 3")
            for i in range(len(d["text"])):
                conf = d["conf"][i]
                word = d["text"][i].strip()
                if not word or not isinstance(conf, (int,float)) or conf < 20: continue
                word_boxes.append({"text": word, "x": d["left"][i], "y": d["top"][i],
                                   "w": d["width"][i], "h": d["height"][i], "conf": float(conf)})
            vc = [b["conf"] for b in word_boxes]
            avg_conf = float(np.mean(vc)) if vc else 50.0
        except Exception: pass

        amounts = []
        for m in re.findall(r"(?:rs\.?\s*|RS\s*|₹\s*)?([\d,]+\.?\d*)", text, re.I):
            try:
                v = float(re.sub(r"[,\s]", "", m))
                if 1.0 < v < 1_000_000: amounts.append(v)
            except ValueError: pass
        amounts = sorted(set(amounts))

        dates = re.findall(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b", text)

        # FIX 2: try single-line match first, then two-line fallback
        total   = None
        _total_re = re.compile(
            r"(?:gr(?:and)?\s*total|total\s*amt?|net\s*payable|amount\s*due|total)"
            r"[:\s\.\(RS₹]*([0-9,]+\.?\d*)", re.I)
        m1 = _total_re.search(text)
        if m1:
            try: total = float(re.sub(r"[,\s]", "", m1.group(1)))
            except ValueError: pass

        if total is None:
            # Try on joined text (keyword + amount may be on adjacent lines)
            joined_for_ocr = _join_split_lines(text)
            m2 = _total_re.search(joined_for_ocr)
            if m2:
                try: total = float(re.sub(r"[,\s]", "", m2.group(1)))
                except ValueError: pass

        vdf  = all(bool(re.match(r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}", d)) for d in dates) if dates else True
        hasc = bool(re.search(r"\$|USD|RM|EUR|MYR|RS|Rs\.|INR|₹", text, re.I))
        lc   = len([l for l in text.splitlines() if l.strip()])
        ma   = max(amounts) if amounts else 0.0
        rat  = (total / ma) if (amounts and total and ma > 0) else 1.0

        return {
            "raw_text": text, "word_boxes": word_boxes,
            "amounts": amounts, "dates": dates, "total": total,
            "max_amount": ma, "n_amounts": len(amounts), "n_dates": len(dates),
            "valid_date_fmt": int(vdf), "has_currency": int(hasc), "line_count": lc,
            "total_to_max_ratio": rat,
            "amount_std": float(np.std(amounts)) if len(amounts) > 1 else 0.0,
            "unique_amount_ratio": (len(set(amounts)) / len(amounts)) if amounts else 1.0,
            "has_total_keyword": int(_fuzzy(text, "total", 2)),
            "avg_word_conf": avg_conf,
            "text_density": len(text.replace("\n","")) / max(lc, 1),
            "round_number_ratio": (sum(1 for a in amounts if a % 1 == 0) / len(amounts)
                                   if amounts else 0.0),
        }
    except Exception: return {}


def _physical(img_rgb: np.ndarray) -> tuple:
    flags, score = [], 0.0
    spatial = {"blob_bbox": None, "ela_ratio": 0.0}

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # ── Localized bright patch detection ─────────────────────
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, blur)

    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < (h * w * 0.001) or area > (h * w * 0.15):
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # ❌ REMOVE this restriction
        # if by < h * 0.5:
        #     continue

        region = gray[by:by+bh, bx:bx+bw]

        if np.std(region) < 6:
            continue

        flags.append("Localized tamper patch detected")
        score += 0.45
        spatial["blob_bbox"] = (bx, by, bw, bh)
        break

    # ── Grey/silver overlay patch detection (printed box over total field) ──
    try:
        img_hsv_g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Grey: very low saturation, mid-range value (not white paper, not black ink)
        grey_mask = cv2.inRange(img_hsv_g,
                                np.array([0,   0,  90], dtype=np.uint8),
                                np.array([180, 35, 200], dtype=np.uint8))

        # Exclude near-white (receipt paper) by requiring it's darker than paper
        # Paper is typically V > 210; grey patch is V in 90–200
        # Also exclude near-black (ink): V > 90 already handles that

        # Morphological cleanup
        kg = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 6))
        grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kg)
        grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_OPEN,  kg)

        grey_contours, _ = cv2.findContours(
            grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in grey_contours:
            area = cv2.contourArea(cnt)
            # Must be between 0.3% and 15% of image area
            if area < (h * w * 0.003) or area > (h * w * 0.15):
                continue
            gx, gy, gw, gh = cv2.boundingRect(cnt)
            aspect = gw / max(gh, 1)
            # Printed boxes over totals are wide rectangles
            if aspect < 1.5:
                continue
            # Must be in lower 70% of receipt
            if gy < h * 0.30:
                continue
            region_g = gray[gy:gy+gh, gx:gx+gw]
            region_std = np.std(region_g)
            # Grey patch is uniform (std 5–40); skip noisy regions (text-heavy)
            if region_std < 5 or region_std > 50:
                continue
            flags.append(
                f"Grey overlay patch detected over receipt field "
                f"({gw}x{gh}px at y={gy}) — total value may be obscured or replaced"
            )
            score += 0.60
            if spatial["blob_bbox"] is None:
                spatial["blob_bbox"] = (gx, gy, gw, gh)
            break
    except Exception:
        pass

    # ── Brightness discontinuity ─────────────────────────────
    bm = [float(gray[int(h*i/4):int(h*(i+1)/4), :].mean()) for i in range(4)]
    mjp = max(abs(bm[i+1] - bm[i]) for i in range(3))

    if mjp > 28:
        flags.append(f"Brightness discontinuity ({mjp:.0f})")
        score += 0.20

    # ── ELA ──────────────────────────────────────────────────
    try:
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 75])

        re_img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        ela = cv2.absdiff(gray, re_img).astype(np.float32)

        ela_n = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, eth = cv2.threshold(ela_n, 80, 255, cv2.THRESH_BINARY)

        ela_r = eth.sum() / 255 / (h * w)
        spatial["ela_ratio"] = ela_r

        if ela_r > 0.03:
            flags.append("Compression anomaly (ELA)")
            score += 0.35

    except:
        pass

    return float(min(score, 1.0)), flags, spatial

    # ── Brightness discontinuity ────────────────────────────────────
    bm = [float(gray[int(h*i/4):int(h*(i+1)/4), :].mean()) for i in range(4)]
    mjp = max(abs(bm[i+1] - bm[i]) for i in range(3))

    if mjp > 28:
        flags.append(f"Brightness discontinuity ({mjp:.0f} levels)")
        score += 0.20

    # ── ELA (Error Level Analysis) ──────────────────────────────────
    try:
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 75])

        re_img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        ela = cv2.absdiff(gray, re_img).astype(np.float32)

        ela_n = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, eth = cv2.threshold(ela_n, 80, 255, cv2.THRESH_BINARY)

        ela_r = eth.sum() / 255 / (h * w)
        spatial["ela_ratio"] = ela_r

        if ela_r > 0.05:
            flags.append(f"ELA: high compression residual ({ela_r*100:.1f}%) — possible paste")
            score += 0.35

    except Exception:
        pass

    # ── 🔥 Subtle texture / patch detection (FIXED & IMPROVED) ───────
    try:
        gray_f = gray.astype(np.float32)

        blur = cv2.GaussianBlur(gray_f, (15, 15), 0)
        sqr_blur = cv2.GaussianBlur(gray_f**2, (15, 15), 0)
        local_var = sqr_blur - blur**2

        var_norm = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX)

        # 🔥 KEY FIX: improved sensitivity
        smooth_mask = var_norm < 0.04
        smooth_ratio = float(smooth_mask.mean())

        if 0.005 < smooth_ratio < 0.25:
            flags.append(f"Subtle texture inconsistency ({smooth_ratio*100:.2f}% smooth area)")
            score += 0.25

            # extra boost for small localized patches
            if smooth_ratio < 0.08:
                score += 0.15

    except Exception:
        pass

    return float(min(score, 1.0)), flags, spatial

# ─────────────────────────────────────────────────────────────────────────────
#  FIX 4 — _text(): all regexes run on joined text
# ─────────────────────────────────────────────────────────────────────────────

def _text(text: str, ocr_fields=None) -> tuple:
    """
    v4.1: all regex checks now run on _join_split_lines(text) so that
    keyword-on-one-line / value-on-next-line formats are handled correctly.
    """
    if not text or len(text.strip()) < 10: return 0.0, []
    flags, score = [], 0.0

    joined = _join_split_lines(text)   # FIX 4

    # Logical / semantic validation (new in v4, fixed inputs in v4.1)
    _oc = ocr_fields if ocr_fields is not None else {}
    if "_logical_score" not in _oc:
        log_score, log_flags = _logical_anomaly_score(text, _oc)
        _oc["_logical_score"] = log_score
        _oc["_logical_flags"] = log_flags
    else:
        log_score = _oc["_logical_score"]
        log_flags = _oc["_logical_flags"]

    score += log_score
    flags.extend(log_flags)

    # Tax math — on joined text (FIX 4)
    sub_m   = re.search(r"sub\s*total\s*[:\-]?\s*(?:rs\.?\s*)?([\d,]+\.?\d*)", joined, re.I)
    grand_m = re.search(r"(?:gr(?:and)?\s*total|total\s*amt?|amount\s*due|net\s*payable)"
                        r"[^:\d]*(?:rs\.?\s*)?([\d,]+)", joined, re.I)
    sgst_m  = re.search(r"s\.?g\.?s\.?t[^:\d]*[\d.]+\s*%[^:\d]*([\d,]+\.?\d*)", joined, re.I)
    cgst_m  = re.search(r"c\.?g\.?s\.?t[^:\d]*[\d.]+\s*%[^:\d]*([\d,]+\.?\d*)", joined, re.I)
    igst_m  = re.search(r"i\.?g\.?s\.?t[^:\d]*[\d.]+\s*%[^:\d]*([\d,]+\.?\d*)", joined, re.I)
    try:
        if sub_m and grand_m:
            sub  = float(re.sub(r"[,\s]","", sub_m.group(1)))
            gst  = float(re.sub(r"[,\s]","", grand_m.group(1)))
            sg   = float(re.sub(r"[,\s]","", sgst_m.group(1))) if sgst_m else 0.0
            cg   = float(re.sub(r"[,\s]","", cgst_m.group(1))) if cgst_m else 0.0
            ig   = float(re.sub(r"[,\s]","", igst_m.group(1))) if igst_m else 0.0
            comp = round(sub + sg + cg + ig, 2)
            if abs(comp - gst) > 2:
                flags.append(f"Tax math mismatch: {sub:.0f}+SGST{sg:.2f}+CGST{cg:.2f}={comp:.2f}, stated={gst:.0f}")
                score += 0.50
    except (ValueError, AttributeError): pass

    # Net payable — on joined text (FIX 4)
    total_m   = re.search(r"total\s+(?:amount|amt)[^:\d]*(?:rs\.?\s*)?([\d,]+\.?\d*)", joined, re.I)
    advance_m = re.search(r"advance[^:\d]*(?:rs\.?\s*)?([\d,]+\.?\d*)", joined, re.I)
    net_m     = re.search(r"net\s+payable[^:\d]*(?:rs\.?\s*)?([\d,]+\.?\d*)", joined, re.I)
    try:
        if total_m and net_m:
            t   = float(re.sub(r"[,\s]","", total_m.group(1)))
            n   = float(re.sub(r"[,\s]","", net_m.group(1)))
            adv = float(re.sub(r"[,\s]","", advance_m.group(1))) if advance_m else 0.0
            if abs(round(t-adv,2) - n) > 2:
                flags.append(f"Net payable math mismatch: {t:.0f}-{adv:.0f}≠{n:.0f}")
                score += 0.55
    except (ValueError, AttributeError): pass

    # Digit prepend/append (on joined)
    try:
        if total_m and net_m:
            ts = re.sub(r"[,.\s]","", total_m.group(1))
            ns = re.sub(r"[,.\s]","", net_m.group(1))
            if len(ns) > len(ts):
                extra = ns[:len(ns)-len(ts)]
                if ns.endswith(ts) and extra.isdigit():
                    flags.append(f"Digit prepended to total: '{ts}'->'{ns}'")
                    score += 0.70
                elif ns.startswith(ts) and ns[len(ts):].isdigit():
                    flags.append(f"Digit appended to total: '{ts}'->'{ns}'")
                    score += 0.70
    except (ValueError, AttributeError, UnboundLocalError): pass

    # Line-item math — per-line, use original text
    lp = re.compile(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)$")
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        m = lp.search(line)
        if m:
            try:
                q,r,a = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if q > 0 and r > 0 and abs(q*r - a) > 1.0:
                    flags.append(f"Line-item math error: {q}x{r}!={a}")
                    score += 0.20
            except ValueError: pass

    # Mixed-case corruption
    if re.search(r"\b(T0tal|T0TAL|TOtal|tOtal|ToTal|TOTL|TOT4L|SUBtotal)\b", text):
        flags.append("Mixed-case/corrupted keyword detected")
        score += 0.30

    return float(min(score, 1.0)), flags


def _cross_signal_reason(img_rgb, seg_mask, ocr_result, phys_spatial) -> tuple:
    """
    Unchanged from v4 logic; benefit from fixed _logical_score in ocr_result.
    """
    flags, score = [], 0.0
    word_boxes = ocr_result.get("word_boxes", [])
    H, W       = img_rgb.shape[:2]
    seg_full   = cv2.resize(seg_mask, (W, H))

    for wb in word_boxes:
        word = wb["text"].lower().strip(":.,()")
        if not any(_fuzzy(word, kw, 1) for kw in CRITICAL_KEYWORDS): continue
        pad = 12
        x1,y1 = max(0, wb["x"]-pad),     max(0, wb["y"]-pad)
        x2,y2 = min(W, wb["x"]+wb["w"]+pad), min(H, wb["y"]+wb["h"]+pad)
        region = seg_full[y1:y2, x1:x2]
        tf = float(region.mean()) if region.size > 0 else 0.0
        if tf > 0.25:
            flags.append(f"Segmentation mask overlaps '{wb['text']}' ({tf*100:.0f}% affected)")
            score += min(0.55 + tf * 0.3, 0.85)
            break

    blob_bbox = phys_spatial.get("blob_bbox")
    if blob_bbox:
        bx, by, bw, bh = blob_bbox
        for wb in word_boxes:
            word = wb["text"].lower().strip(":.,()")
            if not any(_fuzzy(word, kw, 1) for kw in CRITICAL_KEYWORDS): continue
            ox = max(0, min(bx+bw, wb["x"]+wb["w"]) - max(bx, wb["x"]))
            oy = max(0, min(by+bh, wb["y"]+wb["h"]) - max(by, wb["y"]))
            if ox * oy > 0:
                flags.append(f"Physical patch/sticker covers '{wb['text']}' — value may be hidden")
                score += 0.75
                break

    ela_ratio = phys_spatial.get("ela_ratio", 0.0)
    if ela_ratio > 0.04:
        lcc = [wb for wb in word_boxes
               if wb["conf"] < 55 and any(_fuzzy(wb["text"].lower(), kw, 1) for kw in CRITICAL_KEYWORDS)]
        if lcc:
            for wb in lcc:
                flags.append(f"ELA anomaly + low OCR confidence on '{wb['text']}' ({wb['conf']:.0f}%)")
            score += 0.45

    total_words = [wb for wb in word_boxes if _fuzzy(wb["text"].lower(), "total", 1)]
    for tw in total_words:
        same_line = [wb for wb in word_boxes
                     if abs(wb["y"]-tw["y"]) < 20 and wb["x"] > tw["x"]
                     and re.match(r"[\d,]+\.?\d*", wb["text"])]
        if same_line:
            try:
                printed = float(re.sub(r"[,\s]","", same_line[0]["text"]))
                stated  = ocr_result.get("total")
                if stated and abs(printed - stated) > 5:
                    flags.append(f"Total field mismatch: line shows {printed:.0f}, parsed={stated:.0f}")
                    score += 0.35
            except ValueError: pass

    # E. Logical mismatch + spatial signal = strong combined evidence
    logical_score = ocr_result.get("_logical_score", 0.0)
    if logical_score >= 0.30:
        seg_mean = float(seg_full.mean())
        ela_hit  = ela_ratio > 0.04
        blob_hit = blob_bbox is not None

        if seg_mean > 0.02 and logical_score >= 0.55:
            flags.append(
                f"Logical total mismatch ({logical_score:.2f}) + segmentation signal "
                f"({seg_mean*100:.1f}% mask) — strong digital edit evidence")
            score += 0.65
        elif (ela_hit or blob_hit) and logical_score >= 0.30:
            flags.append(
                f"Logical inconsistency ({logical_score:.2f}) corroborated by "
                f"{'ELA' if ela_hit else 'physical patch'} — edit likely")
            score += 0.45
        elif logical_score >= 0.55:
            flags.append(f"Logical total mismatch alone is definitive ({logical_score:.2f})")
            score += 0.40

    return float(min(score, 1.0)), flags


def _decide(clf_prob, seg_score, anom_score, phys_score,
            text_score, cross_score, all_flags, threshold):

    ensemble = min(
        0.25*clf_prob + 0.20*seg_score + 0.15*anom_score +
        0.20*phys_score + 0.15*text_score + 0.25*cross_score, 1.0)

    if phys_score > 0.20 and clf_prob > 0.55:
        return "suspicious", _conf(ensemble, threshold * 0.5)

    # 🔥 Strong signals
    strong = sum([
        seg_score >= 0.35,
        phys_score >= 0.45,
        cross_score >= 0.45,
        clf_prob >= 0.90
    ])

    # 🔥 Moderate signals
    moderate = sum([
        seg_score >= 0.20,
        phys_score >= 0.30,
        anom_score >= 0.30,
        text_score >= 0.25,
        cross_score >= 0.30
    ])

    # ✅ Rule 1: Strong agreement
    if strong >= 2:
        return "forged", _conf(ensemble, threshold)

    # ✅ Rule 2: Multiple moderate signals
    if moderate >= 3:
        return "forged", _conf(ensemble, threshold)

    # ✅ Clean bias (very important)
    if (
        seg_score < 0.12 and
        phys_score < 0.20 and
        anom_score < 0.20 and
        text_score < 0.15 and
        cross_score < 0.15
    ):
        return "clean", _conf(ensemble, threshold * 0.5)

    # ⚠️ Borderline
    if moderate >= 1 or ensemble >= threshold * 0.5:
        return "suspicious", _conf(ensemble, threshold * 0.5)

    return "clean", _conf(ensemble, threshold * 0.5)

def _conf(ensemble, ref) -> float:
    raw = abs(ensemble - ref) / max(ref, 1 - ref)
    return round(min(50.0 + raw * 49.0, 99.0), 1)


def _to_b64(arr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    if not _models: raise HTTPException(503, "Models not loaded.")
    data    = await file.read()
    arr     = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None: raise HTTPException(400, "Could not decode image.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    import torch, torch.nn.functional as F
    device = _models["device"]
    size   = _models["img_size"]

    with torch.no_grad():
        t1 = _preprocess(img_rgb, size).to(device)
        t2 = _preprocess(img_rgb[:,::-1].copy(), size).to(device)
        p1 = F.softmax(_models["clf"](t1), 1)[0,1].item()
        p2 = F.softmax(_models["clf"](t2), 1)[0,1].item()
        clf_prob = (p1 + p2) / 2

        seg_raw     = _models["unet"](t1).squeeze().cpu().numpy()
        H, W        = img_rgb.shape[:2]
        tamper_area = float((seg_raw > 0.35).mean())
        if tamper_area < 0.05:
            seg_score = 0.0
        else:
            seg_score = float(min((tamper_area - 0.05) * 10, 1.0))
        

        hm   = cv2.resize(seg_raw, (W, H))
        hm_u = (hm * 255).clip(0,255).astype(np.uint8)
        hm_c = cv2.cvtColor(cv2.applyColorMap(hm_u, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.55, hm_c, 0.45, 0)

    ocr_result = _ocr(img_rgb)

    # _text() runs first to populate _logical_score cache in ocr_result
    text_score, text_flags = _text(ocr_result.get("raw_text", ""), ocr_result)

    try:
        anom_score, anom_flags = (_models["anom"].score(ocr_result)
                                  if ocr_result else (0.0, []))
    except Exception:
        anom_score, anom_flags = 0.0, []

    phys_score, phys_flags, phys_spatial = _physical(img_rgb)

    seg_mask_full = cv2.resize(seg_raw, (W, H))
    cross_score, cross_flags = _cross_signal_reason(
        img_rgb, seg_mask_full, ocr_result, phys_spatial)

    all_flags = cross_flags + phys_flags + text_flags + anom_flags

    verdict, confidence = _decide(
        clf_prob, seg_score, anom_score,
        phys_score, text_score, cross_score,
        all_flags, _models["threshold"])

    structured_summary = {}
    if "_structured" in ocr_result:
        sf = ocr_result["_structured"]
        structured_summary = {
            "items"           : sf.get("items", [])[:10],
            "taxes"           : sf.get("taxes", [])[:5],
            "discounts"       : sf.get("discounts", [])[:5],
            "subtotal"        : sf.get("subtotal"),
            "total"           : sf.get("total"),
            "inferred_total"  : sf.get("_inferred_total", False),
        }

    return JSONResponse({
        "verdict"    : verdict,
        "is_forged"  : verdict == "forged",
        "confidence" : confidence,
        "clf_prob"   : round(clf_prob    * 100, 1),
        "seg_score"  : round(seg_score   * 100, 1),
        "tamper_area": round(tamper_area * 100, 2),
        "anom_score" : round(anom_score  * 100, 1),
        "phys_score" : round(phys_score  * 100, 1),
        "text_score" : round(text_score  * 100, 1),
        "cross_score": round(cross_score * 100, 1),
        "n_signals"  : sum([seg_score>=0.20, anom_score>=0.25, phys_score>=0.30,
                            text_score>=0.20, cross_score>=0.25]),
        "flags"      : all_flags,
        "ocr": {
            "n_amounts": ocr_result.get("n_amounts", 0),
            "n_dates"  : ocr_result.get("n_dates", 0),
            "total"    : ocr_result.get("total"),
            "avg_conf" : round(ocr_result.get("avg_word_conf", 0), 1),
            "raw_text" : (ocr_result.get("raw_text") or "")[:800],
            "amounts"  : ocr_result.get("amounts", [])[:10],
            "dates"    : ocr_result.get("dates", [])[:5],
        },
        "structured" : structured_summary,
        "overlay_b64": _to_b64(overlay),
    })


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_models)}
