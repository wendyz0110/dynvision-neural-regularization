"""
Overlap between the 1854 THINGS concepts (TVSD) and the 1000 ImageNet-1k classes, via WordNet + manual overrides.

Outputs
  --out           concept level: one row per THINGS concept that overlaps with ImageNet
                    uniqueID         THINGS concept = TVSD folder name = `concept` in the manifest / outputs.npz
                    word             THINGS concept noun
                    n_imagenet       number of matched ImageNet classes
                    imagenet_idx     matched ImageNet class indices (";"-separated), 0-999 as in CordsNet's pred_label
                    imagenet_names   their names
                    source           "wordnet" (automatic) or "manual" (from --overrides)
  --out_images    image level (needs --things_imgs): one row per TVSD stimulus whose concept overlaps
                    split, stim_idx  as in the manifest (stim_idx 1-based = TVSD train_idx / test_idx)
                    concept, things_path, n_imagenet, imagenet_idx, imagenet_names
  --out_full      optional audit table: every concept with its automatic WordNet result (levels below)

Automatic WordNet matching (one THINGS synset per concept):
  match   A        same WordNet synset as an ImageNet class          (zebra = zebra)
          B        ImageNet class(es) are more specific (hyponyms), within --max_down steps
                                                                      (dog -> 118 dog breeds)
          R        the THINGS concept is more specific than an ImageNet class,
                   within --max_up hypernym steps                    (blimp -> airship)
          none     has a WordNet synset, but no A/B/R relation
          no_wordnet   no WordNet synset in the THINGS metadata (e.g. airboat)
  Priority A > B > R: a concept gets the first level that applies. A, B and R all count as overlap.
  review (in --out_full): rows that needed a manual decision:
          "same name, other sense"  word matches an ImageNet class name but WordNet senses differ
          "broad B"                 B match with more than --broad ImageNet classes

Manual overrides (--overrides, default overlap_overrides.csv next to this script), applied last:
  uniqueID, include (1 = overlap with the given imagenet_idx, 0 = not overlap), imagenet_idx, reason
  include = 1 with an empty imagenet_idx keeps the automatic WordNet match (a recorded "checked, keep").

Inputs:
  --things_meta   concepts-metadata_things.tsv (THINGS OSF, jum2f)
  --imagenet      imagenet_wnids.json (next to this script: ImageNet index -> WordNet id -> name;
                  verified identical to Hugging Face ILSVRC/imagenet-1k classes.py)
  --things_imgs   TVSD things_imgs.mat (for --out_images; needs h5py)
Needs nltk with WordNet 3.0:  pip install nltk ; python -c "import nltk; nltk.download('wordnet')"

Usage:
  python build_things_imagenet_overlap.py --things_meta concepts-metadata_things.tsv \
      --out things_imagenet_overlap.csv \
      --things_imgs things_imgs.mat --out_images things_imagenet_overlap_images.csv
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--things_meta", required=True)
ap.add_argument("--imagenet", default=str(Path(__file__).with_name("imagenet_wnids.json")))
ap.add_argument("--overrides", default=str(Path(__file__).with_name("overlap_overrides.csv")))
ap.add_argument("--out", required=True, help="concept-level overlap CSV")
ap.add_argument("--things_imgs", default=None, help="TVSD things_imgs.mat, for the image-level CSV")
ap.add_argument("--out_images", default=None, help="image-level overlap CSV (needs --things_imgs)")
ap.add_argument("--out_full", default=None, help="optional audit table of the automatic WordNet result")
ap.add_argument("--max_up", type=int, default=2, help="max hypernym steps for level R (THINGS more specific)")
ap.add_argument("--max_down", type=int, default=5, help="max hyponym steps for level B (ImageNet more specific)")
ap.add_argument("--broad", type=int, default=20, help="flag B matches with more ImageNet classes than this")
args = ap.parse_args()

# ImageNet classes as WordNet synsets
imagenet = json.load(open(args.imagenet))
syn2idx = {wn.synset_from_pos_and_offset("n", int(c["wnid"][1:])): c["idx"] for c in imagenet}
names = [c["name"] for c in imagenet]
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
imagenet_names_norm = {norm(a) for c in imagenet for a in c["name"].split(",")}

# THINGS concepts
with open(args.things_meta, encoding="utf-8", errors="replace") as fh:
    things = list(csv.DictReader(fh, delimiter="\t"))
assert len(things) == 1854, f"expected 1854 THINGS concepts, got {len(things)}"

rows = []
for t in things:
    row = {"uniqueID": t["uniqueID"], "word": t["Word"], "synset": "", "match": "no_wordnet",
           "n_imagenet": 0, "imagenet_idx": "", "imagenet_names": "", "steps": "",
           "name_match": norm(t["Word"]) in imagenet_names_norm, "review": ""}

    # THINGS synset: sense key (Wordnet ID2) resolves for every concept that has one; fall back to the offset
    syn = None
    key, offset = t["Wordnet ID2"].strip(), t["WordNet ID"].strip()
    if key:
        try:
            syn = wn.lemma_from_key(key).synset()
        except Exception:
            syn = None
    if syn is None and re.fullmatch(r"n\d{8}", offset):
        syn = wn.synset_from_pos_and_offset("n", int(offset[1:]))
    if syn is None:
        rows.append(row)
        continue
    row["synset"], row["match"] = syn.name(), "none"

    # A: same synset
    hits = {syn2idx[syn]: 0} if syn in syn2idx else {}
    level = "A" if hits else None
    # B: ImageNet classes below the THINGS synset, within max_down steps
    if not hits:
        frontier, seen = {syn}, {syn}
        for step in range(1, args.max_down + 1):
            frontier = {h for s in frontier for h in s.hyponyms() + s.instance_hyponyms()} - seen
            seen |= frontier
            hits.update({syn2idx[s]: step for s in frontier if s in syn2idx})
        level = "B" if hits else None
    # R: ImageNet class above the THINGS synset, nearest within max_up steps
    if not hits:
        frontier, seen = {syn}, {syn}
        for step in range(1, args.max_up + 1):
            frontier = {h for s in frontier for h in s.hypernyms() + s.instance_hypernyms()} - seen
            seen |= frontier
            hits = {syn2idx[s]: step for s in frontier if s in syn2idx}
            if hits:
                level = "R"
                break
    if level:
        idx = sorted(hits)
        row.update(match=level, n_imagenet=len(idx), imagenet_idx=";".join(map(str, idx)),
                   imagenet_names=";".join(names[i] for i in idx),
                   steps=";".join(str(hits[i]) for i in idx))
    if row["name_match"] and row["match"] == "none":
        row["review"] = "same name, other sense"
    elif level == "B" and len(hits) > args.broad:
        row["review"] = "broad B"
    rows.append(row)

counts = Counter(r["match"] for r in rows)
print("Automatic WordNet result:", ", ".join(f"{k} {counts[k]}" for k in ["A", "B", "R", "none", "no_wordnet"]),
      f"| string name match alone: {sum(r['name_match'] for r in rows)}")
if args.out_full:
    with open(args.out_full, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# ---- final overlap: automatic A/B/R, then manual overrides ----
overlap = {r["uniqueID"]: (r["imagenet_idx"], "wordnet") for r in rows if r["match"] in ("A", "B", "R")}
by_id = {r["uniqueID"]: r for r in rows}
with open(args.overrides, newline="") as fh:
    overrides = list(csv.DictReader(fh))
for o in overrides:
    assert o["uniqueID"] in by_id, f"override for unknown concept {o['uniqueID']}"
    if o["include"].strip() == "1" and not o["imagenet_idx"].strip():
        assert o["uniqueID"] in overlap, f"{o['uniqueID']}: 'keep' override but no automatic match"
    elif o["include"].strip() == "1":
        overlap[o["uniqueID"]] = (o["imagenet_idx"].strip(), "manual")
    else:
        overlap.pop(o["uniqueID"], None)
flagged = {r["uniqueID"] for r in rows if r["review"]}
undecided = sorted(flagged - {o["uniqueID"] for o in overrides})
if undecided:
    print("WARNING, flagged concepts without a manual decision:", ", ".join(undecided))

final = []
for uid in sorted(overlap):
    idx = [int(i) for i in overlap[uid][0].split(";")]
    final.append({"uniqueID": uid, "word": by_id[uid]["word"], "n_imagenet": len(idx),
                  "imagenet_idx": ";".join(map(str, idx)), "imagenet_names": ";".join(names[i] for i in idx),
                  "source": overlap[uid][1]})
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(final[0]))
    w.writeheader()
    w.writerows(final)
n_manual = sum(f["source"] == "manual" for f in final)
print(f"Overlap: {len(final)} of 1854 THINGS concepts ({len(overrides)} manual decisions, "
      f"{n_manual} added manually) -> {args.out}")

# ---- image level: every TVSD stimulus of an overlapping concept ----
if args.out_images:
    import h5py
    assert args.things_imgs, "--out_images needs --things_imgs"
    concept_info = {f["uniqueID"]: f for f in final}
    img_rows = []
    with h5py.File(args.things_imgs, "r") as f:
        for split, key in [("train", "train_imgs"), ("test", "test_imgs")]:
            paths = ["".join(map(chr, f[r][()].flatten())).replace("\\", "/") for r in f[key]["things_path"][()].flatten()]
            for i, p in enumerate(paths, start=1):
                c = p.split("/")[0]
                assert c in by_id, f"{p}: concept not in THINGS metadata"
                if c in concept_info:
                    info = concept_info[c]
                    img_rows.append({"split": split, "stim_idx": i, "concept": c, "things_path": p,
                                     "n_imagenet": info["n_imagenet"], "imagenet_idx": info["imagenet_idx"],
                                     "imagenet_names": info["imagenet_names"]})
    with open(args.out_images, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(img_rows[0]))
        w.writeheader()
        w.writerows(img_rows)
    sc = Counter(r["split"] for r in img_rows)
    print(f"Images: {sc['train']} train + {sc['test']} test stimuli of overlapping concepts -> {args.out_images}")
