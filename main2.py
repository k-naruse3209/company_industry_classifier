
import asyncio
import aiohttp
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from normalize import normalize_company_name
from sources import wikidata as wd
from sources import wikipedia as wp
from classify import load_wd_to_jsic_map, fallback_from_text
from score import name_similarity, compute_confidence

import argparse
import os

DEFAULT_MAPPING = os.path.join(os.path.dirname(__file__), "mappings", "wd_p452_to_jsic.csv")

async def classify_company(session: aiohttp.ClientSession, name: str, lang_priority: List[str], mapping: dict):
    norm = normalize_company_name(name)
    best = None

    # Step 1: search on Wikidata (ja then en)
    candidates = []
    for lang in lang_priority:
        try:
            res = await wd.wbsearch(session, norm, lang=lang, limit=5)
            if res:
                candidates.extend([(lang, hit) for hit in res])
        except Exception:
            pass

    # Deduplicate by QID while keeping first lang
    seen = set()
    qids = []
    qid_meta = {}
    for lang, hit in candidates:
        qid = hit.get("id")
        if qid and qid not in seen:
            seen.add(qid)
            qids.append(qid)
            qid_meta[qid] = {
                "lang": lang,
                "label": hit.get("label"),
                "description": hit.get("description"),
            }

    if qids:
        entities = await wd.wbgetentities(session, qids, lang="ja")
        entmap = entities.get("entities", {})
    else:
        entmap = {}

    result = {
        "正規化社名": norm,
        "マッチ先": "",
        "業種(原文)": "",
        "業種(JSIC大分類)": "",
        "JSICコード": "",
        "根拠URL1": "",
        "根拠URL2": "",
        "根拠URL3": "",
        "判定スコア": 0.0,
        "判定ステータス": "未判定",
        "補足": "",
    }

    # Score candidates by name similarity (ja label preferred)
    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    for qid in qids:
        entity = entmap.get(qid, {})
        label_ja = wd.get_label(entity, "ja") or qid_meta[qid].get("label")
        sim = name_similarity(norm, (label_ja or ""))
        ranked.append((sim, qid, entity))

    ranked.sort(reverse=True, key=lambda x: x[0])

    had_p452 = False
    had_official = False
    evidence_urls: List[str] = []

    # Try top 3 candidates
    for sim, qid, entity in ranked[:3]:
        p452 = wd.extract_p452_claims(entity)
        label_ja = wd.get_label(entity, "ja") or qid_meta[qid].get("label")
        official = wd.get_official_website(entity)
        if official:
            had_official = True
            evidence_urls.append(official)
        # sitelink -> Wikipedia summary as text fallback context
        title = wd.get_sitelink_title(entity, "jawiki")
        summary_text = ""
        if title:
            try:
                summ = await wp.get_summary(session, title, lang="ja")
                if summ and 'extract' in summ:
                    summary_text = summ['extract']
                    if 'content_urls' in summ and 'desktop' in summ['content_urls']:
                        evidence_urls.append(summ['content_urls']['desktop'].get('page', ""))
            except Exception:
                pass

        jsic_code = jsic_label = ""
        # If P452 present, map each to JSIC
        if p452:
            had_p452 = True
            for ind_qid in p452:
                if ind_qid in mapping:
                    jsic_code, jsic_label = mapping[ind_qid]
                    break  # take first mapped
        # Fallback from summary text
        if not jsic_code:
            fb = fallback_from_text(summary_text or (qid_meta[qid].get("description") or ""))
            if fb:
                jsic_code, jsic_label = fb

        # If we got a JSIC classification, compute confidence and return
        if jsic_code:
            conf = compute_confidence(sim, had_p452, had_official)
            result.update({
                "マッチ先": f"Wikidata:{qid}",
                "業種(原文)": "Wikidata P452" if had_p452 else "Wikipedia summary（キーワード）",
                "業種(JSIC大分類)": jsic_label,
                "JSICコード": jsic_code,
                "判定スコア": conf,
                "判定ステータス": "確定" if conf >= 0.75 else "要確認"
            })
            # Deduplicate evidence urls and trim
            ev = [u for u in dict.fromkeys([u for u in evidence_urls if u])]
            ev += [f"https://www.wikidata.org/wiki/{qid}"]
            ev = ev[:3]
            for i, u in enumerate(ev, start=1):
                result[f"根拠URL{i}"] = u
            return result

    # If no candidate or no classification
    # Try Wikipedia direct search summary (ja) as last resort
    # (omitted: complex search; keep result as 未判定)
    return result

async def process_file(input_csv: str, output_csv: str, company_column: str="B", lang_priority: List[str]=["ja", "en"], mapping_path: str=DEFAULT_MAPPING, concurrency: int=5):
    df = pd.read_csv(input_csv)
    # Determine B column: if header uses letters, or if we need by index
    if company_column in df.columns:
        names = df[company_column].astype(str).fillna("")
    else:
        # Try 1-based Excel-like indexing if "B" specified
        col_idx = 1 if company_column.upper() == "B" else None
        if col_idx is not None and col_idx < len(df.columns):
            names = df.iloc[:, col_idx].astype(str).fillna("")
        else:
            # fallback: first column
            names = df.iloc[:, 0].astype(str).fillna("")

    mapping = load_wd_to_jsic_map(mapping_path)

    results = []
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def sem_task(name):
            async with sem:
                return await classify_company(session, name, lang_priority, mapping)
        tasks = [asyncio.create_task(sem_task(n)) for n in names]
        for coro, name in zip(tasks, names):
            try:
                res = await coro
            except Exception as e:
                res = {
                    "正規化社名": name,
                    "マッチ先": "",
                    "業種(原文)": "",
                    "業種(JSIC大分類)": "",
                    "JSICコード": "",
                    "根拠URL1": "",
                    "根拠URL2": "",
                    "根拠URL3": "",
                    "判定スコア": 0.0,
                    "判定ステータス": "未判定",
                    "補足": f"error: {e}",
                }
            results.append(res)

    # Append columns to original df
    out_df = df.copy()
    res_df = pd.DataFrame(results)
    out_df = pd.concat([out_df, res_df], axis=1)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--company-column", default="B", help="Company column header or letter (default: B)")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--mapping", default=DEFAULT_MAPPING, help="Mapping CSV path")
    args = ap.parse_args()
    asyncio.run(process_file(args.input, args.output, args.company_column, mapping_path=args.mapping, concurrency=args.concurrency))

if __name__ == "__main__":
    main()
