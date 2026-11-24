"""Convert notebooks/forecasting.py (cells marked with `# %%`) into a Jupyter notebook.

Run:  python scripts/build_notebook.py
Then execute it:  jupyter nbconvert --to notebook --execute --inplace notebooks/forecasting.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

SRC = Path(__file__).resolve().parents[1] / "notebooks" / "forecasting.py"
cells = []
for chunk in re.split(r"^# %%", SRC.read_text(), flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    else:
        cells.append(nbf.v4.new_code_cell(body))
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, SRC.with_suffix(".ipynb"))
print(f"{len(cells)} cells -> {SRC.with_suffix('.ipynb')}")
