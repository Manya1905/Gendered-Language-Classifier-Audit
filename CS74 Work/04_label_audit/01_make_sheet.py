"""
E4 step 1: build the blind labeling sheet.

Question: does Claude's labeling match a human's?

Sentences come from the NEW evaluation set, unseen version (flagged only), so every model,
the original included, can later be scored fairly against the human labels:
  - 99 sentences: 33 per Claude label (masculine / feminine / neutral), random within each.
  - 20 extra: sentences Claude labeled neutral but BERT (clean) called masculine in at least
    2 of 3 seeds (the F18 / F23 false alarms). No overlap with the 99.
  - At most 3 sentences per company across the whole sheet. Seed 42.
  - Everything shuffled together.

The sheet shows only what Claude saw: the sentence and its masculine / feminine list words.
No Claude label, reasoning, or model predictions. The answer key (which row is which sentence
and which group) is saved separately in sheet_key.csv: don't open it until labeling is done.

Usage:  python 01_make_sheet.py
Saves:  labeling/blind_labeling_sheet.xlsx, labeling/sheet_key.csv
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "01_clean_reevaluation"
OUT = HERE / "labeling"
KEYS = ["source", "job_id", "sentence_num"]
SEED, PER_CLASS, N_EXTRA, CAP = 42, 33, 20, 3


def load_pool():
    d = pd.read_csv(E1 / "data" / "new_eval.csv", dtype={"job_id": str, "company_id": str})
    d = d[d.flagged & ~d.seen_in_annotated & d.label.notna()].copy()
    # how many of the 3 clean BERT seeds called each sentence masculine
    d["bert_masc_votes"] = 0
    for s in [42, 43, 44]:
        p = pd.read_csv(E1 / "results" / "predictions" / f"bert_seed{s}_new_eval.csv", dtype={"job_id": str})
        m = d[KEYS].merge(p[KEYS + ["pred_id"]], on=KEYS, how="left", validate="one_to_one")
        assert m.pred_id.notna().all()
        d["bert_masc_votes"] += (m.pred_id.values == 1).astype(int)
    return d


def take(pool, n, used_companies):
    """Random rows from pool (seed 42), skipping companies already at the cap."""
    picked = []
    for i, row in pool.sample(frac=1, random_state=SEED).iterrows():
        if used_companies.get(row.company_id, 0) < CAP:
            picked.append(i)
            used_companies[row.company_id] = used_companies.get(row.company_id, 0) + 1
            if len(picked) == n:
                break
    assert len(picked) == n, f"only {len(picked)} of {n} available"
    return picked


def main():
    pool = load_pool()
    used = {}
    chosen = []
    for cls in ["masculine", "feminine", "neutral"]:
        idx = take(pool[pool.label == cls], PER_CLASS, used)
        chosen += [(i, f"claude_{cls}") for i in idx]
    taken = {i for i, _ in chosen}
    extra_pool = pool[(pool.label == "neutral") & (pool.bert_masc_votes >= 2) & ~pool.index.isin(taken)]
    print(f"Pool for the 20 extra (Claude neutral, BERT masculine in >=2 seeds): {len(extra_pool)}")
    chosen += [(i, "bert_false_masculine") for i in take(extra_pool, N_EXTRA, used)]

    sheet = pool.loc[[i for i, _ in chosen]].copy()
    sheet["group"] = [g for _, g in chosen]
    sheet = sheet.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sheet["sheet_id"] = range(1, len(sheet) + 1)

    # checks
    assert len(sheet) == 3 * PER_CLASS + N_EXTRA
    assert not sheet.duplicated(KEYS).any()
    assert sheet.company_id.value_counts().max() <= CAP

    OUT.mkdir(exist_ok=True)
    sheet[["sheet_id", "group"] + KEYS + ["company_id"]].to_csv(OUT / "sheet_key.csv", index=False)
    write_xlsx(sheet, OUT / "blind_labeling_sheet.xlsx")

    print(f"{len(sheet)} sentences, {sheet.company_id.nunique()} companies, "
          f"max {sheet.company_id.value_counts().max()} per company")
    print(sheet.groupby(["group", "industry"]).size().unstack(fill_value=0).to_string())
    print(f"Saved {OUT / 'blind_labeling_sheet.xlsx'} and sheet_key.csv (don't open the key)")


INSTRUCTIONS = """Labeling gendered language in job-posting sentences (E4)

For each sentence on the 'Sentences' tab, pick one label from the dropdown in the Label column.
Each sentence contains at least one word from the gendered word lists; those words are shown next to it,
split into masculine-list and feminine-list words (the same information Claude saw).
Decide whether the flagged word(s) are used in a genuinely gendered way or in a neutral way.

These are the same definitions Claude was given (CS72 prelabel.py prompt, Gaucher et al. 2011):

GENDERED (masculine, feminine, or both) when the word:
- Describes desired candidate traits/qualities ("assertive problem-solver")
- Describes company culture or work environment ("competitive environment", "supportive workplace")
- Describes team values or organizational identity ("we are an ambitious, driven team")
- Describes expected work style ("work independently", "collaborate with the team")
Gendered language is NOT limited to the ideal candidate: descriptions of the company,
culture, and work environment count too ("We are an aggressive, fast-paced company" = masculine).

NEUTRAL only when the word is used in a purely structural, factual, or non-connotative way:
- Compensation/benefits ("competitive salary", "generous benefits")
- Job duty descriptions ("responsible for managing budgets", "lead quarterly reviews")
- Legal/structural terms ("independent contractor", "decision pending approval")
- Non-adjective usage ("kind of work", "force majeure", "yield results")
- Factual market position ("leading provider of healthcare services")

LABELS
masculine - word(s) carry masculine-coded connotation (trait, culture, or work style)
feminine  - word(s) carry feminine-coded connotation (trait, culture, or work style)
both      - masculine AND feminine coded words, both in a gendered context
neutral   - word(s) used in a purely structural/factual/non-connotative way

Use the Notes column for anything you were unsure about (helps the disagreement analysis later).
Judge each sentence on its own. Don't open labeling/sheet_key.csv until you're done."""


def write_xlsx(sheet, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    for line in INSTRUCTIONS.split("\n"):
        ws.append([line])
    ws["A1"].font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 120

    s = wb.create_sheet("Sentences")
    s.append(["ID", "Sentence", "Masculine list words", "Feminine list words", "Label", "Notes"])
    for c in s[1]:
        c.font = Font(bold=True)
    for r in sheet.itertuples():
        s.append([r.sheet_id, r.sentence_text,
                  r.masculine_words if isinstance(r.masculine_words, str) else "",
                  r.feminine_words if isinstance(r.feminine_words, str) else "", None, None])
    for col, w in zip("ABCDEF", [6, 90, 22, 22, 13, 40]):
        s.column_dimensions[col].width = w
    for row in s.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"masculine,feminine,both,neutral"', allow_blank=True)
    s.add_data_validation(dv)
    dv.add(f"E2:E{len(sheet) + 1}")
    s.freeze_panes = "A2"
    wb.active = 1
    wb.save(path)


if __name__ == "__main__":
    main()
