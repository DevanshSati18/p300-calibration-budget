"""Step 10 — screen for the same person appearing in more than one study.

bigP3BCI anonymises EDF headers; some files keep sex and birth year in the patient field
("L_10 M 01-JAN-1965 X_X ALS_36"). Participants in different studies with identical
(sex, birth year) are flagged as POSSIBLE duplicates. A shared birth year is weak evidence
(many participants are students of similar age), so flags are a screen, not an identification.
Output: outputs/tables/participant_overlap.csv
"""
import glob
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p300budget.config import load_config, project_path  # noqa: E402

cfg = load_config()
base = project_path(cfg["paths"]["data_root"])
rows = []
for st in sorted(p for p in os.listdir(base) if p.startswith("Study")):
    for p in sorted(os.listdir(base / st)):
        sex = year = None
        for f in glob.glob(str(base / st / p / "**/*.edf"), recursive=True)[:6]:
            h = open(f, "rb").read(88)[8:88].decode("latin1").split()
            if len(h) >= 3 and h[1] != "X" and not h[2].endswith("2020"):
                sex, year = h[1], h[2][-4:]
                break
        rows.append(dict(study=st[5:], participant=p, sex=sex, birth_year=year))
df = pd.DataFrame(rows)
known = df.dropna(subset=["sex"])
groups = known.groupby(["sex", "birth_year"])
flags = []
for (s, y), g in groups:
    if g.study.nunique() > 1:
        flags.append(dict(sex=s, birth_year=y, participants=" ".join(g.participant), studies=" ".join(sorted(g.study.unique()))))
out = pd.DataFrame(flags)
out.to_csv(project_path(cfg["paths"]["tables"]) / "participant_overlap.csv", index=False)
print(f"participants: {len(df)}; with sex/birth year: {len(known)}; studies without it: "
      f"{sorted(set(df.study) - set(known.study))}")
print(out.to_string(index=False) if len(out) else "no cross-study matches")
