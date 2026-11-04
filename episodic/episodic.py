#!/usr/bin/env python3
"""Retrieval ("episodic memory") baselines for anticipating activity changes on UCI HAR.

Question: can retrieving similar past moments anticipate the activity h windows ahead,
especially where the activity actually changes, better than persistence or Markov?
Does a person's own history help more than other people's? And is retrieval learning
behaviour, or just the fixed order of the data-collection protocol?

Each window is 2.56 s with 50% overlap, so one step ahead = 1.28 s.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ACT = {1: "WALK", 2: "UPSTAIRS", 3: "DOWNSTAIRS", 4: "SIT", 5: "STAND", 6: "LAY"}
ABS_POS = False
METHODS = ["persistence", "markov", "script_position", "population_memory",
           "population_memory_1perperson", "own_history_memory", "combined_memory"]


def load(root):
    root = Path(root)
    out = {}
    for split in ("train", "test"):
        X = np.loadtxt(root / split / f"X_{split}.txt")
        y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)
        s = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        out[split] = (X, y, s)
    return out


def subjects_in_order(s):
    seen = []
    for v in s:
        if v not in seen:
            seen.append(v)
    return seen


def check_order(name, y, s):
    """Rows must be contiguous per subject and in time order for 'h steps ahead' to mean anything."""
    print(f"[order check: {name}]")
    for subj in subjects_in_order(s):
        idx = np.where(s == subj)[0]
        contiguous = bool(np.all(np.diff(idx) == 1))
        changes = int(np.sum(y[idx][1:] != y[idx][:-1]))
        print(f"  subject {subj:2d}: {len(idx):4d} windows, contiguous={contiguous}, label changes={changes}")


def contexts(Z, y, s, h, L):
    """Per subject: context = last L windows (PCA features) up to t; target = label at t+h."""
    out = {}
    for subj in subjects_in_order(s):
        idx = np.where(s == subj)[0]
        n = len(idx)
        ts = np.arange(L - 1, n - h)
        if len(ts) == 0:
            continue
        ctx = np.stack([Z[idx[t - L + 1:t + 1]].ravel() for t in ts])
        out[subj] = dict(rows=idx[ts], t=ts, n=n, pos=(ts.astype(float) if ABS_POS else ts / n), ctx=ctx,
                         cur=y[idx[ts]], fut=y[idx[ts + h]])
    return out


def transition_matrix(y, s, h):
    T = np.zeros((7, 7))
    for subj in subjects_in_order(s):
        ys = y[s == subj]
        for a, b in zip(ys[:-h], ys[h:]):
            T[a, b] += 1
    return T


def vote(dists, labels, k, eps=1e-6):
    order = np.argsort(dists)[:k]
    w = 1.0 / (dists[order] + eps)
    scores = np.zeros(7)
    np.add.at(scores, labels[order], w)
    return int(scores.argmax()), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/UCI HAR Dataset")
    ap.add_argument("--context", type=int, default=4, help="windows of history per context")
    ap.add_argument("--k", type=int, default=10, help="neighbours retrieved")
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--horizons", default="1,2,4,8")
    ap.add_argument("--out", default="results")
    ap.add_argument("--abs-position", action="store_true", help="control uses window index, not t/n")
    a = ap.parse_args()
    global ABS_POS
    ABS_POS = a.abs_position
    L, K = a.context, a.k
    horizons = [int(x) for x in a.horizons.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = load(a.data)
    Xtr, ytr, str_ = d["train"]
    Xte, yte, ste = d["test"]
    check_order("train", ytr, str_)
    check_order("test", yte, ste)

    scaler = StandardScaler().fit(Xtr)
    Str, Ste = scaler.transform(Xtr), scaler.transform(Xte)
    pca = PCA(n_components=a.pca, random_state=0).fit(Str)
    Ztr, Zte = pca.transform(Str), pca.transform(Ste)

    # Current-activity classifier, used by persistence and Markov (no oracle labels for them)
    clf = LogisticRegression(max_iter=3000).fit(Str, ytr)
    cur_pred_rows = clf.predict(Ste)
    print(f"\ncurrent-window classifier accuracy on test subjects: {np.mean(cur_pred_rows == yte):.3f}")

    lines = ["horizon_windows,horizon_s,method,accuracy,transition_accuracy,"
             "transition_acc_subject_min,transition_acc_subject_max,"
             "change_alarms,alarm_precision,n_windows,n_transitions"]
    examples = []

    for h in horizons:
        tr = contexts(Ztr, ytr, str_, h, L)
        te = contexts(Zte, yte, ste, h, L)
        T = transition_matrix(ytr, str_, h)
        pop_ctx = np.vstack([v["ctx"] for v in tr.values()])
        pop_fut = np.concatenate([v["fut"] for v in tr.values()])
        pop_subj = np.concatenate([np.full(len(v["t"]), subj) for subj, v in tr.items()])
        pop_t = np.concatenate([v["t"] for v in tr.values()])
        pop_pos = np.concatenate([v["pos"] for v in tr.values()]).reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=min(len(pop_ctx), K * 40)).fit(pop_ctx)
        nn_pos = NearestNeighbors(n_neighbors=K).fit(pop_pos)

        correct = {m: [] for m in METHODS}
        alarms = {m: [] for m in METHODS}
        per_subj = {m: [] for m in METHODS}
        trans_all = []

        for subj, v in te.items():
            C, ts, fut, cur_true = v["ctx"], v["t"], v["fut"], v["cur"]
            cp = cur_pred_rows[v["rows"]]
            dpop, ipop = nn.kneighbors(C)
            dpos, ipos = nn_pos.kneighbors(v["pos"].reshape(-1, 1))
            D = pairwise_distances(C)
            preds = {m: np.zeros(len(ts), dtype=int) for m in METHODS}

            for i, t in enumerate(ts):
                preds["persistence"][i] = cp[i]
                preds["markov"][i] = int(T[cp[i]].argmax())

                # Control: only "how far into the session" (0-1), no sensor data at all
                preds["script_position"][i], _ = vote(dpos[i], pop_fut[ipos[i]], K, eps=1e-3)

                pp, _ = vote(dpop[i, :K], pop_fut[ipop[i, :K]], K)
                preds["population_memory"][i] = pp

                # K neighbours from K different people (independent votes)
                seen, keep = set(), []
                for j in range(ipop.shape[1]):
                    sj = pop_subj[ipop[i, j]]
                    if sj not in seen:
                        seen.add(sj)
                        keep.append(j)
                    if len(keep) == K:
                        break
                keep = np.array(keep)
                p1, _ = vote(dpop[i, keep], pop_fut[ipop[i, keep]], len(keep))
                preds["population_memory_1perperson"][i] = p1

                # Own history: only past windows whose future label was observed before
                # the current context starts (no overlap, no peeking ahead)
                mask = (ts + h) <= (t - L)
                op = vote(D[i, mask], fut[mask], K)[0] if mask.sum() >= K else pp
                preds["own_history_memory"][i] = op

                cand_d = np.concatenate([dpop[i, :K], D[i, mask]])
                cand_l = np.concatenate([pop_fut[ipop[i, :K]], fut[mask]])
                cand_src = ([("population", int(pop_subj[j]), int(pop_t[j])) for j in ipop[i, :K]]
                            + [("own", int(subj), int(ts[j])) for j in np.where(mask)[0]])
                cbp, corder = vote(cand_d, cand_l, K)
                preds["combined_memory"][i] = cbp

                is_trans = fut[i] != cur_true[i]
                if (h == 4 and is_trans and p1 == fut[i] and cp[i] != fut[i] and len(examples) < 6):
                    ex = [f"subject {subj}, window {t} ({100 * v['pos'][i]:.0f}% into session): now "
                          f"{ACT[cur_true[i]]}, actual in {h * 1.28:.1f} s {ACT[fut[i]]}; persistence "
                          f"{ACT[cp[i]]}, script-position {ACT[preds['script_position'][i]]}, "
                          f"memory (1 per person) {ACT[p1]}",
                          "  retrieved moments (subject, window, distance, what happened next):"]
                    for j in keep[:5]:
                        g = ipop[i, j]
                        ex.append(f"    subject {pop_subj[g]:2d} window {pop_t[g]:4d}  "
                                  f"dist {dpop[i, j]:7.2f}  -> {ACT[pop_fut[g]]}")
                    examples.append("\n".join(ex))

            trans = fut != cur_true
            trans_all.append(trans)
            for m in METHODS:
                c = preds[m] == fut
                correct[m].append(c)
                alarms[m].append(preds[m] != cp)   # method predicts "the activity will change"
                if trans.sum() > 0:
                    per_subj[m].append(c[trans].mean())

        tr_cat = np.concatenate(trans_all)
        print(f"\n=== horizon {h} windows ({h * 1.28:.1f} s): {tr_cat.sum()} transition moments "
              f"of {len(tr_cat)} ===")
        print(f"{'method':<30}{'accuracy':>9}{'trans acc':>11}{'subj range':>14}"
              f"{'alarms':>8}{'alarm prec':>12}")
        for m in METHODS:
            c = np.concatenate(correct[m])
            al = np.concatenate(alarms[m])
            ta = c[tr_cat].mean()
            ps = np.array(per_subj[m])
            prec = (al & tr_cat & c).sum() / al.sum() if al.sum() else float("nan")
            print(f"{m:<30}{c.mean():>9.3f}{ta:>11.3f}{f'{ps.min():.2f}-{ps.max():.2f}':>14}"
                  f"{int(al.sum()):>8}{prec:>12.3f}")
            lines.append(f"{h},{h * 1.28:.2f},{m},{c.mean():.4f},{ta:.4f},{ps.min():.4f},"
                         f"{ps.max():.4f},{int(al.sum())},{prec:.4f},{len(c)},{int(tr_cat.sum())}")

    (out / "episodic_results.csv").write_text("\n".join(lines) + "\n")
    (out / "episodic_examples.txt").write_text("\n\n".join(examples) + "\n")
    print(f"\nwrote {out / 'episodic_results.csv'} and {out / 'episodic_examples.txt'}")
    print("alarm precision = of the windows where a method predicts a change, the share where the\n"
          "activity really changes AND the method named the right new activity")


if __name__ == "__main__":
    main()
