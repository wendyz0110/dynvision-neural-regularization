"""
Check cka_estimators.py against the official code of Chun et al. (2025).

Needs jax and a clone of github.com/badooki/CKA:
  git clone https://github.com/badooki/CKA.git
  python test_cka_estimators.py --reference CKA/CCN_code
For several sizes it compares naive / Song / corrected H values and the three CKAs
computed the way the paper's notebooks do, and checks CKA_0 against plain linear CKA.
"""
import argparse
import sys

import numpy as np
import torch

from cka_estimators import cka_all, h_cross, h_self, linear_cka, moments

ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True, help="folder with the official moment_utils.py")
args = ap.parse_args()
sys.path.insert(0, args.reference)
from moment_utils import getest_all  # noqa: E402

rng = np.random.default_rng(0)
worst = 0.0
for P, Qa, Qb in [(64, 256, 448), (64, 5000, 256), (20, 7, 30), (200, 50, 50), (64, 3000, 3000)]:
    # correlated populations with uneven feature scales and nonzero means
    Z = rng.normal(size=(P, 40))
    A = Z @ rng.normal(size=(40, Qa)) * rng.uniform(0.2, 3, Qa) + rng.normal(size=Qa)
    B = Z @ rng.normal(size=(40, Qb)) + 0.5 * rng.normal(size=(P, Qb)) + 2.0

    ref_num = [float(v) for v in getest_all(A, B, indep_cols=True)]
    ref_da = [float(v) for v in getest_all(A, A, indep_cols=False)]
    ref_db = [float(v) for v in getest_all(B, B, indep_cols=False)]
    ref_cka = [ref_num[0] / np.sqrt(ref_da[0] * ref_db[0]),
               ref_num[1] / np.sqrt(ref_da[1] * ref_db[1]),
               ref_num[1] / np.sqrt(ref_da[2] * ref_db[2])]

    ma, mb = moments(torch.from_numpy(A)), moments(torch.from_numpy(B))
    ours_num = [float(v) for v in h_cross(ma, mb)]
    ours_da = [float(v) for v in h_self(ma)]
    ours_db = [float(v) for v in h_self(mb)]
    ours_cka = [float(v) for v in cka_all(ma, mb)]
    plain = float(linear_cka(torch.from_numpy(A), torch.from_numpy(B)))

    pairs = list(zip(ref_num[:2], ours_num)) + list(zip(ref_da, ours_da)) + list(zip(ref_db, ours_db)) \
        + list(zip(ref_cka, ours_cka)) + [(ours_cka[0], plain)]
    rel = max(abs(r - o) / max(abs(r), 1e-300) for r, o in pairs)
    worst = max(worst, rel)
    print(f"P={P:4d} Qa={Qa:5d} Qb={Qb:5d}  CKA_0 {ours_cka[0]:.6f}  CKA_S {ours_cka[1]:.6f}  "
          f"CKA_C {ours_cka[2]:.6f}  (official {ref_cka[2]:.6f})  max rel. diff {rel:.1e}")

print("max relative difference over all checks:", f"{worst:.1e}")
assert worst < 1e-8, "port does not match the official implementation"
print("OK: port matches the official implementation")
