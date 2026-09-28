#!/usr/bin/env python3
"""Select the lowest-development-loss completed training seed."""
import glob
import json
import sys

run_dir, output_path = sys.argv[1:]
rows = []
for path in sorted(glob.glob(run_dir + "/seed*/manifest.json")):
    with open(path) as stream:
        data = json.load(stream)
    if data.get("status") != "complete":
        raise SystemExit(f"incomplete training manifest: {path}")
    rows.append((float(data["best_dev_soft_ce"]), path, int(data["seed"])))
if len(rows) != 4:
    raise SystemExit(f"expected four completed seeds, found {len(rows)}")
rows.sort()
selection = {
    "selected_seed": rows[0][2],
    "selection": "minimum dev soft cross-entropy",
    "best_dev_soft_ce": rows[0][0],
    "manifest": rows[0][1],
    "all_seeds": [
        {"seed": row[2], "best_dev_soft_ce": row[0]} for row in rows
    ],
}
with open(output_path, "w") as stream:
    json.dump(selection, stream, indent=2)
    stream.write("\n")
print(json.dumps(selection, indent=2))
