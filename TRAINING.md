# How the model is trained (and how to make it accurate)

Written for someone who has not trained a model before. Everything below refers
to this project's actual files.

## 1. What "training a model" actually means

You are not teaching a computer to understand canals. You are doing something
much narrower and more mechanical:

> You have a table. Some rows have an answer written next to them. The training
> algorithm finds a rule that reproduces those answers from the other columns.
> Then you apply that rule to the rows that have no answer yet.

That's it. A model is a rule fitted to examples.

For AYACUT:

- **The table** is `data/features.csv` — one row per sample point along a canal,
  4,445 rows, produced by step 2.
- **The columns** (the *features*) are 13 numbers per point, all from
  Sentinel-2: NDVI on the channel, NDVI in the ring beside it, the difference
  between them, MNDWI, water fraction — for a wet-season and a dry-season
  window, plus three seasonal-contrast features.
- **The answer** (the *label*) is what you type in step 3: `flowing`, `dry`,
  `choked`, or `encroached`. You will label about 300 of the 4,445 rows.
- **Training** takes those ~300 labelled rows and fits a rule.
- **Prediction** applies that rule to all 4,445 rows, including the ~4,145 you
  never looked at.

The reason this is worth doing: you can label 300 points in an hour. You cannot
label 4,445. And a real department cannot inspect 900 km of canal every season.

## 2. What the rule looks like here

We use a **random forest**. It is a collection of decision trees, each of which
is a nested set of yes/no questions:

```
is dry_ndvi_diff > 0.12 ?
├── yes: is wet_water_frac < 0.05 ?
│        ├── yes  -> choked
│        └── no   -> flowing
└── no:  is dry_mndwi > -0.1 ?
         ├── yes  -> flowing
         └── no   -> dry
```

One tree alone is unstable — change a few labels and it looks completely
different. So we grow 400 trees, each on a random subset of the data and
features, and let them vote. That averaging is what makes it robust.

**Why not a neural network / deep learning:**

1. You will have ~300 labelled examples. Deep models need thousands to tens of
   thousands. With 300 they memorise and fail on anything new.
2. You must explain every prediction to judges and eventually to an engineer
   who will send a crew somewhere. A random forest can tell you "this reach was
   flagged because its dry-season channel NDVI was 0.31 above the surrounding
   fields". A neural net cannot.
3. The features already encode the physics. The hard thinking went into
   *designing* `ndvi_diff` — vegetation in the channel relative to the cropland
   beside it. Once you have the right feature, the classifier's job is easy.
   This is the single most important idea in the project.

## 3. Train/test split — and the mistake almost everyone makes

If you fit a rule to 300 points and then measure how well it reproduces those
same 300 answers, you learn nothing. It has seen the answers. It will score
high and mean nothing.

So you hold data back. Fit on 75%, measure on the 25% the model never saw.

**The subtle part, and it matters here:** sample points are spaced 200 m apart
along the same canal segment. Two points 200 m apart on the same canal look
nearly identical. If one lands in training and its neighbour lands in test, the
model effectively saw the test answer. Your reported accuracy will be inflated,
and the model will disappoint the moment it meets a canal it hasn't seen.

This is called **leakage**, and it's the most common way hackathon ML results
are quietly wrong.

The fix, already implemented in
[step04_train_classifier.py](step04_train_classifier.py): split by **segment**,
not by point. Every point on a given canal segment goes entirely to training or
entirely to test. `GroupKFold` and `GroupShuffleSplit` with `groups=seg_id` do
this. The accuracy you get will be *lower* than a naive split — and it will be
the true number.

If a judge asks one hard question about your ML, it will probably be this one.
You now have the good answer.

## 4. Reading the output

`python step04_train_classifier.py` prints four things:

**Cross-validation accuracy**, e.g. `0.78 +/- 0.06`. The fraction of held-out
points classified correctly, averaged over 5 different train/test divisions.
The `+/-` matters as much as the number: `0.78 +/- 0.06` is a real signal;
`0.78 +/- 0.25` means your folds disagree and you need more labels.

**Per-class precision and recall.** Accuracy alone lies when classes are
uneven. If 80% of your points are `flowing`, a model that answers "flowing"
every single time scores 80% accuracy and is useless.

- *Precision* for `choked`: of the points the model called choked, what
  fraction really were? Low precision → you send crews to healthy canals.
- *Recall* for `choked`: of the points that really were choked, what fraction
  did the model find? Low recall → you miss blocked canals.

For this project **recall on `choked` matters more than precision.** Missing a
blocked canal that serves 4,000 ha is far worse than sending an inspector to a
canal that turns out to be fine.

**Confusion matrix.** Rows are truth, columns are prediction. The diagonal is
correct. Off-diagonal cells tell you *which* mistakes it makes — confusing
`dry` with `choked` is a different problem from confusing `dry` with `flowing`.

**Feature importances.** Two versions are printed. Gini importance is fast but
biased toward high-variance features. Permutation importance is the honest one:
it shuffles a feature and measures how much accuracy drops. Quote permutation
importance to judges.

If `ndvi_diff` tops that list, your core hypothesis is confirmed by data — say
so out loud in the demo.

## 5. How to actually get it accurate

In rough order of payoff:

**Label carefully, and label the hard cases.** The single biggest lever. 300
sloppy labels beat nothing but lose badly to 300 careful ones. `step03` already
samples in a stratified way across canal type and NDVI difference, so you see
the full range rather than 300 near-identical healthy reaches.

**Use the `unclear` button.** If you can't tell from the chip, press `unclear`
— those rows are dropped from training. A guessed label is worse than no label,
because it teaches the model a wrong rule.

**Watch class balance.** If you end up with 250 `flowing` and 6 `choked`, the
model cannot learn `choked`. The script warns when the rarest class is under 5.
The fix is to deliberately go find more choked examples, not to change the
model. `class_weight="balanced"` is already set, which helps but does not
substitute for real examples.

**Label in pairs.** Two teammates label the same 30 points independently and
compare. If you agree on only 60%, your *labels* are the accuracy ceiling —
no model can beat the consistency of its training data. Fix the definitions
first, then label the rest.

**Then, and only then, tune the model.** Changing `n_estimators` or `max_depth`
will move accuracy a percent or two. Getting 100 more good labels will move it
ten. Beginners tune first because it feels like progress; it isn't.

## 6. What to say when a judge pushes on accuracy

Don't claim a number you can't defend. Say this instead:

> "We report segment-grouped cross-validation accuracy, so no canal appears in
> both training and test. Our headline metric is recall on the impaired classes,
> because the cost of missing a blocked canal that serves 4,000 hectares is much
> higher than the cost of an unnecessary inspection. The classifier outputs a
> probability, not a verdict — it produces a ranked inspection list, and a PWD
> engineer confirms on the ground."

That framing is both more honest and more convincing than a big accuracy
number, because it shows you understand what the model is for.

## 7. The honest limit

The model sees the surface. It cannot see the canal bed. A channel full of
water looks the same from orbit whether the bed is clean or half full of silt.

What it genuinely detects is vegetation growing in a channel, and channels that
stay dry when they should be carrying water — both real, visible consequences
of a canal that has stopped conveying properly.

So the product is **a prioritised inspection list**, not a silt survey. That is
still enormously useful: it turns "we have 900 km and a complaints register"
into "here are the 20 reaches to look at first, ranked by how much farmland
depends on them." Never oversell it past that — a judge who knows remote
sensing will catch it, and the honest version is a stronger pitch anyway.
