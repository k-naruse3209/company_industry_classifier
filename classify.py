
from typing import List, Optional, Tuple
import re
import csv
import os

# Load mapping from Wikidata industry QID -> JSIC code (and label)
def load_wd_to_jsic_map(path: str):
    mp = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wd = row.get("wikidata_qid","").strip()
                jsic_code = row.get("jsic_code","").strip()
                jsic_label = row.get("jsic_label","").strip()
                if wd and jsic_code:
                    mp[wd] = (jsic_code, jsic_label)
    return mp

# Simple keyword -> JSIC mapping (fallback when WD P452 not present)
FALLBACK_RULES = [
    (re.compile(r"\b(retail|supermarket|convenience store|百貨店|小売)\b", re.I), ("I", "卸売業・小売業")),
    (re.compile(r"\b(wholesale|卸売)\b", re.I), ("I", "卸売業・小売業")),
    (re.compile(r"\b(software|saas|it services|情報サービス|ソフトウェア)\b", re.I), ("G-39", "情報サービス業")),
    (re.compile(r"\b(telecom|telecommunications|通信|携帯電話)\b", re.I), ("G", "情報通信業")),
    (re.compile(r"\b(bank|銀行|credit|金融|fintech)\b", re.I), ("J", "金融業・保険業")),
    (re.compile(r"\b(manufactur|factory|製造|メーカー)\b", re.I), ("E", "製造業")),
    (re.compile(r"\b(construction|建設)\b", re.I), ("D", "建設業")),
    (re.compile(r"\b(hotel|宿泊|旅館)\b", re.I), ("M", "宿泊業、飲食サービス業")),
    (re.compile(r"\b(food service|restaurant|飲食|外食)\b", re.I), ("M-76/77", "飲食店/持ち帰り・配達")),
]

def fallback_from_text(text: str) -> Optional[Tuple[str,str]]:
    if not text:
        return None
    for pat, jsic in FALLBACK_RULES:
        if pat.search(text):
            return jsic
    return None
