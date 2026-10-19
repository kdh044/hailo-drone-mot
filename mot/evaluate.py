import sys, os, glob, contextlib, io, json
import trackeval
from common import ROOT, GROUPS

# usage: python evaluate.py [tracker-name substring ...]  -> results.json + summary table
# tracker results live at data/trackers/VD<group>-val/<name>/data/<seq>.txt
filters = sys.argv[1:]
EVAL_GROUPS = list(GROUPS) + (["all"] if os.path.isdir(f"{ROOT}/data/gt/VDall-val") else [])  # all = 사람+차 통합
results = json.load(open(f"{ROOT}/results.json")) if os.path.exists(f"{ROOT}/results.json") else {}
for g in EVAL_GROUPS:
    bench = f"VD{g}"
    names = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/data/trackers/{bench}-val/*"))
    names = [n for n in names if not filters or n in filters or any(f.endswith("*") and n.startswith(f[:-1]) for f in filters)]
    if not names: continue
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                PLOT_CURVES=False, PRINT_ONLY_COMBINED=True, USE_PARALLEL=True, NUM_PARALLEL_CORES=4,
                TIME_PROGRESS=False)
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=f"{ROOT}/data/gt", TRACKERS_FOLDER=f"{ROOT}/data/trackers",
                SEQMAP_FOLDER=f"{ROOT}/data/seqmaps", BENCHMARK=bench, SPLIT_TO_EVAL="val",
                TRACKERS_TO_EVAL=names, DO_PREPROC=False, PRINT_CONFIG=False)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = trackeval.Evaluator(ecfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)], metrics)
    for n, r in res["MotChallenge2DBox"].items():
        out = {}
        for seq, sr in r.items():
            p = sr["pedestrian"]
            out[seq] = {k: float(v) for k, v in {
                "HOTA": p["HOTA"]["HOTA"].mean() * 100, "DetA": p["HOTA"]["DetA"].mean() * 100,
                "AssA": p["HOTA"]["AssA"].mean() * 100, "LocA": p["HOTA"]["LocA"].mean() * 100,
                "DetRe": p["HOTA"]["DetRe"].mean() * 100, "DetPr": p["HOTA"]["DetPr"].mean() * 100,
                "MOTA": p["CLEAR"]["MOTA"] * 100, "IDF1": p["Identity"]["IDF1"] * 100,
                "IDSW": p["CLEAR"]["IDSW"], "FP": p["CLEAR"]["CLR_FP"], "FN": p["CLEAR"]["CLR_FN"],
                "Frag": p["CLEAR"]["Frag"]}.items()}
        results.setdefault(n, {})[g] = out
json.dump(results, open(f"{ROOT}/results.json", "w"), indent=1)

cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]
for g in EVAL_GROUPS:
    print(f"\n[{g}] COMBINED")
    print(f"{'tracker':32s}" + "".join(f"{c:>8s}" for c in cols))
    rows = [(n, r[g]["COMBINED_SEQ"]) for n, r in results.items() if g in r and (not filters or n in filters or any(f.endswith("*") and n.startswith(f[:-1]) for f in filters))]
    for n, m in sorted(rows, key=lambda x: -x[1]["HOTA"]):
        print(f"{n:32s}" + "".join(f"{m[c]:8.1f}" if c not in ("IDSW", "FP", "FN") else f"{int(m[c]):8d}" for c in cols))
