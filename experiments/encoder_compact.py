"""Print encoder results without policy weights, one JSON object per line, between markers.

  python experiments/encoder_compact.py DIR          # in CI: puts results into the job log
  python experiments/encoder_compact.py --parse LOG  # turn a saved job log back into JSON lines
"""
import glob, json, sys
from pathlib import Path

DROP = ("policy", "trace", "test_detail", "sensor", "n_in", "steps", "val", "x_res_label")
RND = lambda v: round(v, 3) if isinstance(v, float) else v

if sys.argv[1] == "--parse":
    for line in open(sys.argv[2]):
        if "ENC>" in line:
            print(line.split("ENC>", 1)[1].strip())
    sys.exit()
for p in sorted(glob.glob(str(Path(sys.argv[1]) / "*.json*"))):
    rows = [json.loads(l) for l in open(p)] if p.endswith(".jsonl") else [json.load(open(p))]
    for r in rows:
        print("ENC>" + json.dumps({k: RND(v) for k, v in r.items() if k not in DROP or (k == "sensor" and isinstance(v, dict))}, separators=(",", ":")))
