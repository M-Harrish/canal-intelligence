"""Step 6: budget allocation -> which segments to desilt first.

Compares two ways of spending the same rupees:

  AYACUT      greedy knapsack maximising hectares-protected per rupee
  Complaint   status quo proxy: work goes where complaints are loudest, which
              tracks visible neglect and how many people live nearby, not how
              much command area is at stake

Headline number is the comparison: hectares protected per rupee, both ways,
same budget.

Run:
  python step06_allocate_budget.py                 # default budget
  python step06_allocate_budget.py --budget 50000000
"""

import argparse
import pickle
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

import assumptions
import config

DEFAULT_BUDGET = 50_000_000  # Rs 5 crore


class SupplyModel:
    """Hectares protected, without double counting.

    A field is watered only if EVERY canal between it and the anicut conveys,
    so a segment's local area is supplied with probability

        P(supplied) = prod of health(segment) along its supply path

    and hectares at risk = sum of local_area * (1 - P(supplied)).

    Summing `command_area` instead would double count — a parent's already
    contains its children's. Local area plus path products counts each hectare
    once and can never exceed what the network commands.

    It also gets the operational point right: fixing one link only helps if the
    others work. Desilting a clean main above a blocked distributary buys
    nothing.
    """

    def __init__(self, df, graph_pkl):
        with open(graph_pkl, "rb") as f:
            D = pickle.load(f)

        self.seg_ids = list(df["seg_id"])
        self.idx = {s: i for i, s in enumerate(self.seg_ids)}
        n = len(self.seg_ids)

        self.local = df["local_area_ha"].to_numpy(float)
        self.health = df["health_score"].to_numpy(float).clip(0.01, 1.0)
        self.cost = df["cost_inr"].to_numpy(float)

        # path_mask[i, j] = 1 if canal segment j lies on the supply path of i
        self.path_mask = np.zeros((n, n))
        edge_of = {d["seg_id"]: (u, v) for u, v, d in D.edges(data=True)}
        seg_at = {(u, v): d["seg_id"] for u, v, d in D.edges(data=True)}
        sources = [n_ for n_, d in D.nodes(data=True) if d.get("is_source")]

        for sid, i in self.idx.items():
            self.path_mask[i, i] = 1.0
            if sid not in edge_of:
                continue
            u, _ = edge_of[sid]
            for src in sources:
                if not nx.has_path(D, src, u):
                    continue
                nodes = nx.shortest_path(D, src, u)
                for a, b in zip(nodes[:-1], nodes[1:]):
                    up = seg_at.get((a, b))
                    if up in self.idx:  # rivers/connectors treated as working
                        self.path_mask[i, self.idx[up]] = 1.0
                break

        self.baseline_at_risk = self.at_risk(self.health)

    def at_risk(self, health):
        p_supplied = np.exp(self.path_mask @ np.log(health))
        return float((self.local * (1 - p_supplied)).sum())

    def protected(self, chosen_idx):
        h = self.health.copy()
        if len(chosen_idx):
            h[list(chosen_idx)] = np.maximum(
                h[list(chosen_idx)], config.DESILT_TARGET_HEALTH)
        return self.baseline_at_risk - self.at_risk(h)

    def marginal(self, chosen_idx, cand):
        return self.protected(list(chosen_idx) + [cand]) - self.protected(chosen_idx)


def greedy_knapsack(model, budget):
    """Each round, take the affordable segment with the best MARGINAL hectares
    per rupee given what is already chosen. Recomputing the margin every round
    is what stops it buying a parent and its child for the same benefit."""
    chosen, spent = [], 0.0
    remaining = set(range(len(model.seg_ids)))
    while True:
        best, best_ratio = None, 0.0
        for c in remaining:
            if spent + model.cost[c] > budget:
                continue
            ratio = model.marginal(chosen, c) / max(model.cost[c], 1.0)
            if ratio > best_ratio:
                best, best_ratio = c, ratio
        if best is None:
            break
        chosen.append(best)
        spent += model.cost[best]
        remaining.discard(best)
    return chosen, spent


def complaint_baseline(df, model, budget, rng):
    """Proxy for how desilting lists actually get built: squeaky wheel first.

    Complaints rise with visible degradation and with how many people live
    near the canal (segment length, in a delta this settled), but not with
    command area. That mismatch is what AYACUT is for.
    """
    weight = (1 - df["health_score"]).to_numpy() * np.sqrt(
        df["length_m"].to_numpy())
    weight = np.maximum(weight, 1e-9)
    p = weight / weight.sum()

    order = rng.choice(len(df), size=len(df), replace=False, p=p)
    chosen, spent = [], 0.0
    for i in order:
        if spent + model.cost[i] <= budget:
            chosen.append(int(i))
            spent += model.cost[i]
    return chosen, spent


def summarize(name, df, model, chosen, spent, budget):
    ha = model.protected(chosen)
    km = df.iloc[chosen]["length_m"].sum() / 1000
    print(f"\n{name}")
    print(f"  segments selected : {len(chosen)}")
    print(f"  length            : {km:,.1f} km")
    print(f"  spent             : Rs {spent:,.0f} of Rs {budget:,.0f} "
          f"({spent/budget*100:.1f}%)")
    print(f"  hectares protected: {ha:,.0f} ha")
    if spent > 0:
        print(f"  efficiency        : {ha/spent*1e6:,.1f} ha per Rs 10 lakh")
    return ha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help="available desilting budget in rupees")
    ap.add_argument("--trials", type=int, default=200,
                    help="random trials for the complaint-driven baseline")
    args = ap.parse_args()

    if not config.RANKED_GPKG.exists():
        print("no ranked_segments.gpkg — run step05_rank_segments.py first")
        return 1

    df = gpd.read_file(config.RANKED_GPKG).reset_index(drop=True)
    df["cost_inr"] = [
        assumptions.cost_to_desilt(r.length_m, r.can_type)
        for r in df.itertuples()
    ]
    model = SupplyModel(df, config.GRAPH_PKL)

    assumptions.print_all()
    print(f"\nnetwork: {len(df)} segments, {df['length_m'].sum()/1000:,.0f} km, "
          f"full desilt would cost Rs {df['cost_inr'].sum():,.0f}")
    print(f"command area: {model.local.sum():,.0f} ha")
    print(f"currently at risk: {model.baseline_at_risk:,.0f} ha "
          f"({model.baseline_at_risk/model.local.sum()*100:.0f}% of command area)")
    print(f"budget: Rs {args.budget:,.0f} "
          f"({args.budget/df['cost_inr'].sum()*100:.1f}% of the network)")

    ay_chosen, ay_spent = greedy_knapsack(model, args.budget)
    ay_ha = summarize("AYACUT (marginal-benefit greedy knapsack)",
                      df, model, ay_chosen, ay_spent, args.budget)

    rng = np.random.default_rng(config.LABEL_SEED)
    trials, spends = [], []
    for _ in range(args.trials):
        ch, sp = complaint_baseline(df, model, args.budget, rng)
        trials.append(model.protected(ch))
        spends.append(sp)
    trials, spends = np.array(trials), np.array(spends)

    print(f"\nComplaint-driven baseline ({args.trials} random trials)")
    print(f"  spent (mean)      : Rs {spends.mean():,.0f} of "
          f"Rs {args.budget:,.0f}")
    print(f"  hectares protected: {trials.mean():,.0f} ha "
          f"(median {np.median(trials):,.0f}, "
          f"p10 {np.percentile(trials, 10):,.0f}, "
          f"p90 {np.percentile(trials, 90):,.0f})")
    print(f"  efficiency        : "
          f"{trials.mean()/max(spends.mean(), 1)*1e6:,.1f} ha per Rs 10 lakh")

    factor = ay_ha / trials.mean() if trials.mean() > 0 else float("inf")
    print("\n" + "=" * 72)
    print(f"IMPROVEMENT FACTOR: {factor:.2f}x")
    print(f"  AYACUT protects {ay_ha:,.0f} ha for the same budget that the "
          f"complaint-driven\n  approach spends to protect {trials.mean():,.0f} ha "
          f"on average.")
    print(f"  Extra hectares protected: {ay_ha - trials.mean():,.0f} ha")
    beat = (trials < ay_ha).mean() * 100
    print(f"  AYACUT beats {beat:.0f}% of complaint-driven draws.")
    print("=" * 72)

    out = df.iloc[ay_chosen].copy()
    out["selected_by"] = "ayacut_greedy"
    out["work_order"] = np.arange(1, len(out) + 1)  # greedy selection order
    marginal, running = [], []
    for k in range(len(ay_chosen)):
        marginal.append(model.protected(ay_chosen[:k + 1])
                        - model.protected(ay_chosen[:k]))
        running.append(model.protected(ay_chosen[:k + 1]))
    out["ha_protected_marginal"] = marginal
    out["ha_protected_cumulative"] = running
    path = config.DATA_DIR / "work_plan.gpkg"
    out.to_file(path, driver="GPKG")
    print(f"\nwrote {path} ({len(out)} segments)")

    cols = ["work_order", "rank", "can_name", "can_type", "length_m",
            "command_area_ha", "health_score", "cost_inr",
            "ha_protected_marginal"]
    print("\nwork plan (greedy selection order — do these first):")
    print(out[cols].head(15).to_string(index=False, formatters={
        "length_m": "{:,.0f}".format, "command_area_ha": "{:,.0f}".format,
        "health_score": "{:.2f}".format, "cost_inr": "{:,.0f}".format,
        "ha_protected_marginal": "{:,.0f}".format}))

    pd.DataFrame({
        "metric": ["budget_inr", "ayacut_ha", "complaint_ha_mean",
                   "improvement_factor", "ayacut_segments"],
        "value": [args.budget, ay_ha, trials.mean(), factor, len(ay_chosen)],
    }).to_csv(config.DATA_DIR / "budget_comparison.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
