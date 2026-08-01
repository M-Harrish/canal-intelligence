

import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

import config

S2_FEATURES = [
    "wet_ndvi_chan", "wet_ndvi_ring", "wet_ndvi_diff", "wet_mndwi", "wet_water_frac",
    "dry_ndvi_chan", "dry_ndvi_ring", "dry_ndvi_diff", "dry_mndwi", "dry_water_frac",
    "ndvi_diff_amp", "water_frac_drop", "ndvi_chan_amp",
]

# DW class probabilities over the corridor. NDVI can't separate a tree from a
# paddy crop; DW's trees/built bands line up with "choked" and "encroached".
DW_FEATURES = [
    "wet_dw_water", "wet_dw_trees", "wet_dw_crops", "wet_dw_built",
    "wet_dw_shrub_and_scrub", "wet_dw_bare", "wet_dw_flooded_vegetation",
    "dry_dw_water", "dry_dw_trees", "dry_dw_crops", "dry_dw_built",
    "dry_dw_shrub_and_scrub", "dry_dw_bare", "dry_dw_flooded_vegetation",
    "dw_water_drop", "dw_woody", "dw_crops_amp",
]

FEATURES = S2_FEATURES  # extended with DW_FEATURES when dw_features.csv exists


def load_features():
    global FEATURES
    df = pd.read_csv(config.FEATURES_CSV).drop_duplicates("point_id")
    # seasonal contrast: a healthy channel swings, a choked one tracks the land
    df["ndvi_diff_amp"] = df["wet_ndvi_diff"] - df["dry_ndvi_diff"]
    df["water_frac_drop"] = df["wet_water_frac"] - df["dry_water_frac"]
    df["ndvi_chan_amp"] = df["wet_ndvi_chan"] - df["dry_ndvi_chan"]

    if config.DW_FEATURES_CSV.exists():
        dw = pd.read_csv(config.DW_FEATURES_CSV).drop_duplicates("point_id")
        df = df.merge(dw, on="point_id", how="left")
        # woody cover: the clearest sign of a section lost to vegetation
        df["dw_woody"] = df["dry_dw_trees"] + df["dry_dw_shrub_and_scrub"]
        df["dw_water_drop"] = df["wet_dw_water"] - df["dry_dw_water"]
        df["dw_crops_amp"] = df["wet_dw_crops"] - df["dry_dw_crops"]
        FEATURES = S2_FEATURES + DW_FEATURES
        print(f"loaded {len(df)} points with Dynamic World features "
              f"({len(FEATURES)} features)")
    else:
        print(f"loaded {len(df)} points, Sentinel-2 only "
              f"({len(FEATURES)} features) — run step02b for Dynamic World")
    return df


def train(df):
    labels = pd.read_csv(config.LABELS_CSV).drop_duplicates("point_id", keep="last")
    labels = labels[labels["label"] != "unclear"]
    data = df.merge(labels[["point_id", "label"]], on="point_id")
    data = data.dropna(subset=FEATURES + ["label"])
    print(f"{len(data)} labelled points on {data['seg_id'].nunique()} segments")
    print(data["label"].value_counts().to_string())

    counts = data["label"].value_counts()
    if len(counts) < 2:
        print("\nERROR: need at least 2 classes to train")
        return None
    if counts.min() < 5:
        print(f"\nWARNING: rarest class has {counts.min()} examples — "
              f"metrics will be unstable")

    X, y, groups = data[FEATURES].values, data["label"].values, data["seg_id"].values

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    n_groups = data["seg_id"].nunique()
    n_splits = min(5, n_groups, int(counts.min()))
    if n_splits >= 2:
        scores = cross_val_score(clf, X, y, groups=groups,
                                 cv=GroupKFold(n_splits=n_splits), n_jobs=-1)
        print(f"\nsegment-grouped {n_splits}-fold CV accuracy: "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    clf.fit(X[tr], y[tr])
    print(f"\nheld-out test ({len(te)} points, "
          f"{len(set(groups[te]))} unseen segments):")
    print(classification_report(y[te], clf.predict(X[te]), zero_division=0))
    print("confusion matrix (rows=true, cols=pred), classes:", list(clf.classes_))
    print(confusion_matrix(y[te], clf.predict(X[te]), labels=clf.classes_))

    perm = permutation_importance(clf, X[te], y[te], n_repeats=20,
                                  random_state=42, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": FEATURES,
        "gini_importance": clf.feature_importances_,
        "permutation_importance": perm.importances_mean,
    }).sort_values("permutation_importance", ascending=False)
    print("\nfeature importances:")
    print(imp.to_string(index=False))

    clf.fit(X, y)  # refit on everything for scoring
    with open(config.MODEL_PKL, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURES}, f)
    print(f"\nwrote {config.MODEL_PKL}")
    return clf


def score_points(df, clf):
    ok = df.dropna(subset=FEATURES).copy()
    proba = clf.predict_proba(ok[FEATURES].values)
    classes = list(clf.classes_)
    for i, c in enumerate(classes):
        ok[f"p_{c}"] = proba[:, i]
    impaired = [c for c in config.IMPAIRED_CLASSES if c in classes]
    ok["p_impaired"] = ok[[f"p_{c}" for c in impaired]].sum(axis=1)
    ok["pred_class"] = clf.predict(ok[FEATURES].values)
    ok["method"] = "random_forest"
    print(f"scored {len(ok)} / {len(df)} points "
          f"({len(df) - len(ok)} dropped for missing features)")
    return ok


def score_points_provisional(df):
    """Transparent heuristic used ONLY when no labels exist yet.

    A conveying channel looks different from the fields beside it; one whose
    vegetation tracks the surrounding cropland is suspect.

    The score is a PERCENTILE RANK within this network, not a probability — the
    raw NDVI difference has no natural 0-1 scale, and step 5 only needs a
    relative order. It says "more in-channel vegetation than 90% of the
    network", not "90% likely to be blocked".

    water_frac gets low weight on purpose: a 10 m pixel over a 5 m channel is
    mostly bank, so water rarely shows up even on flowing canals.
    """
    ok = df.dropna(subset=["dry_ndvi_diff", "wet_ndvi_diff"]).copy()

    veg_rank = ok["dry_ndvi_diff"].rank(pct=True)
    amp_rank = (-ok["ndvi_diff_amp"]).rank(pct=True)  # little seasonal swing = suspect
    water_bonus = (ok["wet_water_frac"] > 0.10).astype(float)

    if "dw_woody" in ok.columns and ok["dw_woody"].notna().any():
        # DW separates woody growth from crops, so where it exists it outweighs
        # the index ranks and its water band replaces the crude MNDWI bonus
        woody_rank = ok["dw_woody"].rank(pct=True)
        built_rank = ok["dry_dw_built"].rank(pct=True)
        water_bonus = (ok["wet_dw_water"] > 0.15).astype(float)
        score = (0.30 * veg_rank + 0.20 * amp_rank
                 + 0.35 * woody_rank + 0.15 * built_rank
                 - 0.20 * water_bonus)
    else:
        score = 0.6 * veg_rank + 0.4 * amp_rank - 0.15 * water_bonus
    ok["p_impaired"] = score.clip(0, 1)
    ok["pred_class"] = np.where(ok["p_impaired"] > 0.66, "suspect_provisional",
                                "ok_provisional")
    ok["method"] = "PROVISIONAL_HEURISTIC_NOT_VALIDATED"
    print(f"scored {len(ok)} points with the PROVISIONAL heuristic "
          f"(percentile-rank based)")
    return ok


def aggregate(points):
    """Segment health = 1 - mean P(impaired). The 75th percentile is also kept:
    a mostly-fine segment with one badly blocked reach still fails to convey."""
    g = points.groupby("seg_id")
    seg = g.agg(
        n_points=("p_impaired", "size"),
        mean_p_impaired=("p_impaired", "mean"),
        p75_p_impaired=("p_impaired", lambda s: s.quantile(0.75)),
        worst_p_impaired=("p_impaired", "max"),
    ).reset_index()
    seg["health_score"] = (1 - seg["mean_p_impaired"]).clip(0, 1)
    seg["health_score_worst_reach"] = (1 - seg["p75_p_impaired"]).clip(0, 1)
    seg["dominant_class"] = g["pred_class"].agg(lambda s: s.mode().iloc[0]).values
    seg["method"] = points["method"].iloc[0]
    seg = seg.sort_values("health_score")
    seg.to_csv(config.SEGMENT_HEALTH_CSV, index=False)
    print(f"\nwrote {config.SEGMENT_HEALTH_CSV} ({len(seg)} segments)")
    print(f"health score: min {seg.health_score.min():.2f}, "
          f"median {seg.health_score.median():.2f}, max {seg.health_score.max():.2f}")
    return seg


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    df = load_features()
    print(f"{len(df)} feature rows loaded")

    if mode == "provisional":
        print("\n" + "=" * 68)
        print("PROVISIONAL MODE — heuristic score, NOT a trained or validated model.")
        print("Run step 3 labelling, then rerun without 'provisional'.")
        print("=" * 68 + "\n")
        points = score_points_provisional(df)
    else:
        if not config.LABELS_CSV.exists():
            print("no data/labels.csv — run step03_label_points.py, or use:\n"
                  "  python step04_train_classifier.py provisional")
            return 1
        clf = train(df)
        if clf is None:
            return 1
        points = score_points(df, clf)

    points.to_csv(config.POINT_PRED_CSV, index=False)
    print(f"wrote {config.POINT_PRED_CSV}")
    aggregate(points)


if __name__ == "__main__":
    sys.exit(main())
