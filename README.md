# Company Industry Classifier (B列の社名→業種/JSIC 自動判定)

## 概要
CSVのB列にある社名をWikidata / Wikipediaから自動同定し、Wikidataの業種プロパティ(P452)や本文のキーワードから**日本標準産業分類(JSIC)**に正規化して追記します。

## セットアップ
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方
```bash
python main.py --input "/mnt/data/クエリ2 (2)-表1.csv" --output "/mnt/data/output_with_industry.csv" --company-column B
```
- `--company-column` は列名またはExcel風の列記号（例: `B`）。
- 併せて `/mappings/wd_p452_to_jsic.csv` を必要に応じて拡充してください。

## 出力列
- 正規化社名 / マッチ先 / 業種(原文) / 業種(JSIC大分類) / JSICコード / 根拠URL1..3 / 判定スコア / 判定ステータス / 補足

## 注意
- APIレート制限により実行時間は件数に依存します。429/503時は指数バックオフで自動再試行します。
- Wikipediaのインフォボックス解析ではなく、Wikidata P452を優先します。
