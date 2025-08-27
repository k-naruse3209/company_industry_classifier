#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
業種判定プログラム（会社公式サイト本文の自動判定：NLP/ルール併用）

使い方（例）:
  python main.py \
    --input "/path/to/input.csv" \
    --output "/path/to/output_with_industry.csv" \
    --company-column B \
    --concurrency 5

オプション:
  --zero-shot         : 多言語NLIモデル（XLM-RoBERTa-XNLI）でゼロショット判定を併用（重い）
  --rules rules.yaml  : ルール辞書を外部YAMLから読み込み（未指定時は組み込み辞書を使用）
  --debug             : デバッグログ表示

要件（pip）:
  pip install pandas httpx[http2] beautifulsoup4 lxml trafilatura ddgs janome rapidfuzz pyyaml
  # ゼロショットを使う場合
  pip install transformers torch sentencepiece

入力CSV:
  - 会社名がある列（例: B列）を --company-column で指定
  - 既に公式サイトURL列がある場合は自動検出（列名に 'url','URL','website','サイト' が含まれる列）

出力CSV:
  以下の列を追加します：
    found_url, industry_top, score, method, second_best, third_best, matched_keywords, text_sample

注意:
  - 実運用では検索APIの安定性確保（レート制御・リトライ）、robots/ToS遵守、
    タイムアウト・キャッシュ、ユーザーエージェント設定を適切に行ってください。
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import httpx
from bs4 import BeautifulSoup

# 軽量で高精度な本文抽出
import trafilatura

# 公式サイト探索（DuckDuckGo）
from ddgs import DDGS

# 日本語形態素（純Pythonで環境依存少）
from janome.tokenizer import Tokenizer

# 文字列類似度
from rapidfuzz import fuzz

# ルール定義の外部化
import yaml

# ゼロショット（任意）
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ==============================
# 組み込みの業種ルール辞書（初期版）
# - キーワードが見つかるとスコア加点
# - 正規表現もOK（例: r"\\bAI\\b"）
# - 必要に応じて YAML で差し替え可能
# ==============================
DEFAULT_RULES: Dict[str, Dict] = {
    "IT・ソフトウェア": {
        "keywords": [
            "ソフトウェア", "システム開発", "受託開発", "受託システム", "Webシステム", "アプリ開発",
            "SIer", "SES", "ITコンサル", "クラウド", "SaaS", "API", "AI", "機械学習",
            "データ分析", "DX", "インフラ構築", "ネットワーク構築", "保守運用", "基幹システム",
            "ERP", "Salesforce", "kintone", "RPA", "ECサイト", "フロントエンド", "バックエンド",
        ],
        "weight": 1.0,
    },
    "製造": {
        "keywords": [
            "製造", "工場", "加工", "射出成形", "金型", "部品", "組立", "OEM", "品質管理",
            "材料", "製品開発", "量産", "自動化設備",
        ],
        "weight": 1.0,
    },
    "卸売・商社": {
        "keywords": ["卸売", "商社", "仕入", "輸入", "輸出", "販売代理店", "ディストリビュータ"],
        "weight": 1.0,
    },
    "小売・EC": {
        "keywords": ["小売", "EC", "通販", "ネットショップ", "店舗", "百貨店", "ドラッグストア"],
        "weight": 1.0,
    },
    "建設・建築": {
        "keywords": [
            "建設", "建築", "土木", "設計", "施工", "設備工事", "リフォーム", "外構", "解体",
            "住宅", "マンション", "戸建", "ゼネコン",
        ],
        "weight": 1.0,
    },
    "不動産": {
        "keywords": ["不動産", "賃貸", "売買", "管理会社", "仲介", "物件", "テナント", "土地活用"],
        "weight": 1.0,
    },
    "医療・介護": {
        "keywords": [
            "医療", "病院", "クリニック", "薬局", "調剤", "介護", "看護", "福祉", "訪問介護",
            "施設", "デイサービス",
        ],
        "weight": 1.0,
    },
    "飲食・食品": {
        "keywords": ["飲食", "レストラン", "カフェ", "居酒屋", "フード", "食品製造", "惣菜", "ベーカリー"],
        "weight": 1.0,
    },
    "物流・運輸": {
        "keywords": ["物流", "倉庫", "配送", "運送", "輸送", "宅配", "フォークリフト", "保管"],
        "weight": 1.0,
    },
    "教育・人材": {
        "keywords": ["教育", "学校", "塾", "研修", "人材紹介", "人材派遣", "採用支援"],
        "weight": 1.0,
    },
    "金融・保険": {
        "keywords": ["金融", "銀行", "保険", "証券", "ファイナンス", "リース", "与信"],
        "weight": 1.0,
    },
    "農林水産": {
        "keywords": ["農業", "畜産", "水産", "漁業", "養殖", "農園", "栽培"],
        "weight": 1.0,
    },
    "観光・宿泊": {
        "keywords": ["ホテル", "旅館", "観光", "ツアー", "旅行", "民泊", "観光業"],
        "weight": 1.0,
    },
    "広告・マーケティング": {
        "keywords": ["広告", "マーケティング", "PR", "ブランディング", "SNS運用", "SEO", "SEM"],
        "weight": 1.0,
    },
    "その他サービス": {
        "keywords": ["コンサルティング", "清掃", "警備", "レンタル", "BPO", "アウトソーシング"],
        "weight": 0.6,
    },
}

# 公式サイトらしさを判定する語（リンク抽出時にも利用）
ABOUT_LINK_HINTS = [
    "会社概要", "企業情報", "事業内容", "事業案内", "サービス", "製品", "ソリューション",
    "沿革", "ご挨拶", "代表挨拶", "会社情報", "Corporate", "About", "Service", "Product",
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
)

# ============ ユーティリティ ============

def column_to_index(col: str) -> int:
    """Excel風の列記号（A,B,C, ...）を0始まりのindexへ"""
    col = col.strip().upper()
    if not col.isalpha():
        raise ValueError(f"Invalid column: {col}")
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ============ 本文抽出 ============
async def fetch(client: httpx.AsyncClient, url: str, timeout: float = 20.0) -> Optional[str]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("text"):
            return r.text
    except Exception:
        return None
    return None


def extract_main_text(html: str, url: Optional[str] = None) -> str:
    if not html:
        return ""
    try:
        txt = trafilatura.extract(
            filecontent=html,
            url=url,
            favor_recall=True,  # 会社サイトは情報量が少ない場合があるので広めに取得
        )
        return txt or ""
    except Exception:
        return ""


# ============ 公式サイト探索 ============

def is_likely_official_domain(url: str, company: str) -> int:
    """公式サイトっぽさのスコア（大きいほど公式らしい）"""
    score = 0
    if re.search(r"\.(co|ne|or)\.jp(/|$)", url):
        score += 30
    if url.endswith(".jp"):
        score += 10
    if re.search(r"/company|/about|/corporate|/overview", url):
        score += 5
    # 会社名との一致度（Fuzzy）
    base = re.sub(r"株式会社|有限会社|合同会社|\s|　", "", company)
    dom = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
    score += int(fuzz.partial_ratio(base.lower(), dom.lower()) / 10)
    return score


def pick_best_site(candidates: List[Tuple[str, str]], company: str) -> Optional[str]:
    if not candidates:
        return None
    ranked = sorted(
        ((is_likely_official_domain(u, company), u, t) for u, t in candidates),
        key=lambda x: x[0], reverse=True
    )
    return ranked[0][1]


def search_official_site(company: str, max_results: int = 10) -> Optional[str]:
    query = f"\"{company}\" 会社 公式 サイト"
    cands: List[Tuple[str, str]] = []
    try:
        with DDGS(timeout=15) as ddg:
            for r in ddg.text(query, max_results=max_results, safesearch="off"):  # type: ignore
                url = r.get("href") or r.get("url")
                title = r.get("title", "")
                if not url:
                    continue
                # 明らかにSNS・求人・地図は除外
                if re.search(r"(facebook|twitter|x\.com|instagram|linkedin|wantedly|indeed|recruit|google\.com/maps)", url):
                    continue
                cands.append((url, title))
    except Exception:
        pass
    return pick_best_site(cands, company)


# ============ 日本語トークナイズ & ルールスコア ============
JANOME = Tokenizer()


def extract_keywords(text: str) -> List[str]:
    """名詞・カタカナ語を中心に取り出す（簡易）。"""
    kws: List[str] = []
    for token in JANOME.tokenize(text):
        base = token.base_form if token.base_form != "*" else token.surface
        pos = token.part_of_speech.split(',')[0]
        if pos in ("名詞", "形容動詞語幹") and len(base) >= 2:
            kws.append(base)
    return kws


def score_by_rules(text: str, rules: Dict[str, Dict]) -> Tuple[str, float, List[Tuple[str, float]], List[str]]:
    text_lc = text  # 日本語は小文字化の恩恵が小さい
    matched: Dict[str, float] = {k: 0.0 for k in rules.keys()}
    matched_terms: List[str] = []
    for label, spec in rules.items():
        weight = float(spec.get("weight", 1.0))
        for pat in spec.get("keywords", []):
            try:
                if pat.startswith('r"') or pat.startswith("r'"):
                    # 明示的な正規表現。r"..." or r'...'
                    m = re.findall(pat[2:-1], text_lc)
                    if m:
                        matched[label] += weight * len(m)
                        matched_terms.extend([f"{label}:{pat}"] * len(m))
                else:
                    cnt = text_lc.count(pat)
                    if cnt:
                        matched[label] += weight * cnt
                        matched_terms.extend([f"{label}:{pat}"] * cnt)
            except Exception:
                continue
    ranked = sorted(matched.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = ranked[0]
    return top_label, top_score, ranked, matched_terms


# ============ ゼロショット併用（任意） ============
class ZeroShot:
    def __init__(self, labels: List[str]):
        if pipeline is None:
            raise RuntimeError("transformers が未インストールです。pip install transformers torch sentencepiece")
        self.clf = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=-1,
        )
        self.labels = labels

    def classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not text.strip():
            return "", 0.0, {}
        out = self.clf(text[:4000], candidate_labels=self.labels, multi_label=False)
        # out: {'labels': [..], 'scores':[...], 'sequence': '...'}
        labels = out["labels"]
        scores = out["scores"]
        dist = {l: float(s) for l, s in zip(labels, scores)}
        top = labels[0]
        return top, float(scores[0]), dist


# ============ リンク探索（会社概要/事業内容など） ============

def find_about_links(html: str, base_url: str, limit: int = 6) -> List[str]:
    urls: List[str] = []
    try:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            text = (a.get_text() or "").strip()
            href = a["href"].strip()
            if any(h in text for h in ABOUT_LINK_HINTS) or any(h.lower() in href.lower() for h in ["company", "about", "corporate", "service", "product", "business"]):
                urls.append(httpx.URL(href, base=base_url).human_repr())
    except Exception:
        pass
    # 重複除去
    seen = set()
    filtered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            filtered.append(u)
    return filtered[:limit]


# ============ 会社ごとの判定処理 ============
@dataclasses.dataclass
class CompanyResult:
    company: str
    url: str
    industry_top: str
    score: float
    method: str
    second_best: str
    third_best: str
    matched_keywords: str
    text_sample: str


async def classify_company(company: str, client: httpx.AsyncClient, rules: Dict[str, Dict], zs: Optional[ZeroShot]) -> CompanyResult:
    url = search_official_site(company)
    page_html = await fetch(client, url) if url else None

    # 代表ページが取れない場合は早期リターン
    if not page_html:
        return CompanyResult(company, url or "", "判定不可", 0.0, "no_site", "", "", "", "")

    # 会社概要/事業内容ページも辿る
    links = find_about_links(page_html, url)
    htmls = [page_html]

    tasks = [fetch(client, u) for u in links]
    for coro in asyncio.as_completed(tasks):
        h = await coro
        if h:
            htmls.append(h)
        if len(htmls) >= 4:  # 取りすぎ防止
            break

    # 本文抽出をまとめて
    texts = [extract_main_text(h, url) for h in htmls]
    all_text = normalize_space("\n".join([t for t in texts if t]))

    if not all_text:
        return CompanyResult(company, url, "判定不可", 0.0, "no_text", "", "", "", "")

    # ルール判定
    r_label, r_score, ranked, matched_terms = score_by_rules(all_text, rules)

    top, score, method = r_label, float(r_score), "rule"

    # ゼロショット併用（あればスコアとラベルをブレンド）
    if zs is not None:
        z_label, z_prob, z_dist = zs.classify(all_text)
        # 単純加重平均：rule はスケール未校正なので非線形性回避のため正規化
        r_norm = r_score / max(r_score, 10.0)
        z_norm = z_prob
        blend = 0.5 * r_norm + 0.5 * z_norm
        if z_prob >= 0.60 and z_label != r_label:
            top, score, method = z_label, z_prob, "zero-shot"
        else:
            top, score, method = r_label, blend, "rule+zs"

    # 2位3位
    second = ranked[1][0] if len(ranked) > 1 else ""
    third = ranked[2][0] if len(ranked) > 2 else ""

    sample = all_text[:220]

    return CompanyResult(
        company=company,
        url=url or "",
        industry_top=top,
        score=round(float(score), 4),
        method=method,
        second_best=second,
        third_best=third,
        matched_keywords=", ".join(sorted(set(matched_terms))[:20]),
        text_sample=sample,
    )


# ============ メイン ============
async def main_async(args):
    # ルールの準備
    rules = DEFAULT_RULES
    if args.rules:
        with open(args.rules, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                rules = loaded

    zs = None
    if args.zero_shot:
        try:
            zs = ZeroShot(labels=list(rules.keys()))
        except Exception as e:
            logging.warning("ゼロショット初期化に失敗: %s", e)
            zs = None

    df = pd.read_csv(args.input)

    # URL列の推定（任意）
    url_col = None
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["url", "website", "site", "サイト"]):
            url_col = c
            break

    comp_idx = column_to_index(args.company_column)
    company_names: List[str] = []
    for i, row in df.iterrows():
        try:
            company_names.append(str(row.iloc[comp_idx]).strip())
        except Exception:
            company_names.append("")

    # HTTPクライアント
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency)
    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(20.0, connect=10.0)

    async with httpx.AsyncClient(http2=True, limits=limits, headers=headers, timeout=timeout, follow_redirects=True) as client:
        sem = asyncio.Semaphore(args.concurrency)

        async def run_one(i: int, company: str) -> CompanyResult:
            if not company or company.lower() in ("nan", "none"):
                return CompanyResult(company or "", "", "判定不可", 0.0, "no_company", "", "", "", "")
            async with sem:
                return await classify_company(company, client, rules, zs)

        tasks = [run_one(i, name) for i, name in enumerate(company_names)]
        results: List[CompanyResult] = []
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
            except Exception as e:
                logging.exception("分類中に例外: %s", e)
                res = CompanyResult("", "", "判定不可", 0.0, "exception", "", "", "", "")
            results.append(res)

    # 出力
    # 既存DFに列追加
    out = df.copy()
    # 位置は末尾に追加
    out["found_url"] = [r.url for r in results]
    out["industry_top"] = [r.industry_top for r in results]
    out["score"] = [r.score for r in results]
    out["method"] = [r.method for r in results]
    out["second_best"] = [r.second_best for r in results]
    out["third_best"] = [r.third_best for r in results]
    out["matched_keywords"] = [r.matched_keywords for r in results]
    out["text_sample"] = [r.text_sample for r in results]

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="会社公式サイト本文からの業種自動判定（NLP+ルール）")
    p.add_argument("--input", required=True, help="入力CSVパス")
    p.add_argument("--output", required=True, help="出力CSVパス")
    p.add_argument("--company-column", required=True, help="会社名の列（例: B）")
    p.add_argument("--concurrency", type=int, default=5, help="同時並列数")
    p.add_argument("--zero-shot", action="store_true", help="ゼロショット分類を併用する")
    p.add_argument("--rules", default=None, help="YAML形式のルール辞書パス")
    p.add_argument("--debug", action="store_true", help="デバッグログ")
    return p.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
