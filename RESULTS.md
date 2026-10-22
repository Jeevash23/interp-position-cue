# AI-08 results

GPT-2 Medium, indirect-object sentences, inference only. The fixed signal is the clean-minus-corrupted output of twelve
attention heads (6.4, 6.15, 7.2, 7.4, 7.11, 8.15, 9.3, 10.1, 10.12, 11.6, 11.14, 12.14) at the repeated name of a source
sentence, cached once and added at the repeated-name position of corrupted recipient sentences. The *margin* is the
logit of the intended answer minus that of the repeated subject. *Matched* sources name the two people in the
recipient's order; *opposite* sources in the reverse order. The *order contrast* D is the margin with the matched
signal minus the margin with the opposite signal. Brackets are crossed-bootstrap intervals (name units and places
resampled independently, 20,000 draws) at 95% unless a prespecified family level is given. **Post hoc** marks numbers
computed after the studies ended, from saved outputs, with each study's own resampling scheme and seed. Every number
below is one of the values that `paper/make_numbers.py` generates from saved outputs for the manuscript, and
`make check-paper` checks that each number in this file matches one of them.

## Study 1 — one gain per context, or a change of direction? (`studies/fixed-signal-v1`)

- **Primary test (prespecified).** On the confirmation set (12 name groups × 12 places, 1,152 recipients), a
  directional response model improved on one gain per context by **0.0378 [0.0091, 0.0664]** in normalised squared
  error (0.343 → 0.306), **below the prespecified 0.05**, so the planned causal stage was not run. Relative
  improvement 11.0% [2.8, 18.3] (post hoc interval). All other criteria passed: perpendicular improvement 0.0305
  [0.0088, 0.0508]; positive in 10 of 12 name groups and 12 of 12 places; clean accuracy at least 97.2% in every cell.
- **Pilot holdout.** Improvement 0.0548 [0.0223, 0.0922], but one wording-and-order cell failed the clean-accuracy check
  (27/32 = 84.4% against 85%).
- **Caveats of the directional model.** The selected rank (8) was the largest in the search grid; strength +1 carries
  90.5% of the target energy that weights the loss; the map improves error at +0.5 (0.083) and +1 (0.035) but worsens
  it at −0.5 (−0.037).
- **Name order decides the sign of the behavioural effect (secondary stratification; intervals post hoc).** At positive
  strengths the matched source raised the margin by **+1.27 [1.10, 1.47]** logits (top-1 accuracy 34.8% → 76.9%) and
  the opposite source lowered it by **1.11 [0.99, 1.25]** (accuracy 14.6%); 99.3% of opposite-order edits lowered the
  margin. Pooled over both orders the effect was only +0.079 [0.040, 0.122]. At strength +1 the order contrast was
  3.75 [3.32, 4.24] logits.

## Study 2 — a position cue read through name-mover queries (`studies/position-routing-v2`)

Designed after Study 1 to test its order split. Confirmation: 8 new name units × 8 new places, 512 recipients; every
recipient also received *foreign* sources whose names share nothing with its own. Candidate readers: the five GPT-2
Medium name-mover heads 14.14, 15.14, 16.15, 17.4 and 19.15 of Merullo et al. (ICLR 2024); no head search.

- **The effect is positional.** D = **4.09 [3.41, 4.80]** logits, positive in all 8 name units and all 8 places. With
  foreign sources **100.2% [92.7, 108.8]** of it remained (98.33% interval). Matched +2.17, opposite −1.92 logits;
  top-1 44.3% → 96.1% and 7.2%. Unnormalised signals gave D = 4.35.
- **The queries carry part of it.** Holding the five heads' final-position queries at their unedited values removed
  **46.2% [43.5, 50.4]** of D; inserting the edited queries alone recreated **31.3% [25.4, 35.9]** (98.33% intervals).
  Attention probabilities 46.8% [44.2, 50.6], head outputs 46.2% [43.5, 50.0], keys 2.4% [0.9, 5.0], values −0.9%
  [−1.1, −0.7], queries of other heads in the same layers −0.2% [−0.6, 0.2] (95%, post hoc).
- **Criterion.** Retention passed; removal and recreation fell below the prespecified 50%, so the criterion failed. The
  rotation control fell just short as well (excess 24.7% [19.4, 29.2] against the required 25%).
- **Both orders, same queries (post hoc).** The query clamp removed 44.6% [43.1, 46.3] of the matched benefit and 48.1%
  [43.9, 55.4] of the opposite harm; the queries alone recreated 30.2% [24.7, 34.8] and 32.4% [28.5, 35.9].

## Study 3 — two more heads, one MLP, and how they combine (`studies/pathway-interactions-v3`)

Designed after Study 2 to find what carries the rest of the order contrast. A development screen (32 cases) held the
five queries fixed and clamped each of 284 final-position components (267 heads, 17 MLP blocks in blocks 7–23); a rule
fixed in advance selected heads **15.4** and **12.3** (16.2% each) and **MLP block 14** (5.0%); next were heads 14.5
(11.5%) and 18.5 (6.2%). Confirmation: 8 new name triples × 8 new places × 2 new wordings, both orders and roles, two
corruptions (third name; reversed role) at strengths 0.5 and 1: 1,024 cases, 86 conditions each.

| Setting | D (logits) | Q removal | Q+A+M removal | Q recreation | Q+A+M recreation | Joint − sum |
|---|---|---|---|---|---|---|
| Third name, 0.5 | 0.82 | 42.6 [36.6, 47.4] | 72.2 [66.3, 77.1] | 41.1 [34.4, 46.1] | 70.3 [64.3, 74.9] | −24.3 [−28.0, −19.8] |
| Third name, 1.0 | 2.95 | 47.1 [42.8, 50.6] | **70.9 [66.1, 75.1]** | 34.7 [30.7, 38.4] | **63.4 [58.7, 67.5]** | −16.0 [−20.0, −11.6] |
| Reversed role, 0.5 | 1.59 | 55.1 [48.6, 62.2] | 86.7 [79.4, 94.8] | 60.6 [52.7, 69.2] | 77.7 [71.4, 83.8] | −23.7 [−27.6, −19.2] |
| Reversed role, 1.0 | 4.35 | 36.5 [31.9, 40.1] | 71.9 [66.5, 75.6] | 51.7 [47.2, 55.8] | 72.1 [68.6, 75.2] | −28.0 [−31.4, −23.9] |

Percent of D. Q = the five queries; A = heads 15.4 and 12.3; M = MLP block 14. Q intervals 95% (in the saved summary
but absent from the study's own report); Q+A+M intervals are the prespecified 99.6875%; joint − sum 95%.

- **Criterion.** Joint removal and recreation of at least 75% (with extra effect beyond Q of at least 15%) held in 1
  of 4 settings; MLP 14 missed its 5% specificity margin in 3 of 4. The prespecified criterion failed.
- **What adds.** The two heads beat their same-layer control heads by 21.3–34.2% of D for removal and 15.9–26.3% for
  recreation; MLP 14 beat its control by −0.05 to 5.03% and −0.84 to 5.15%. The attention-only subset Q+A (one of the
  prespecified factorial conditions) removed 68.3% [65.0, 71.2] and recreated 60.9% [58.0, 63.5] at third name, 1.0.
- **Sub-additive.** The joint transplant recreated 16.0–28.0 percentage points of D less than the sum of the three
  separate transplants.
- **Development was optimistic.** Joint removal fell from development to confirmation: 105.0% → 86.7% and 91.7% →
  71.9% (reversed role), 81.2% → 72.2% and 79.9% → 70.9% (third name).
- **Same components for benefit and harm (post hoc).** At third name, 1.0, Q+A+M removed 69.5% [66.3, 72.6] of the
  matched benefit and 72.6% [68.9, 75.9] of the opposite harm, and recreated 62.0% [58.3, 65.3] and 65.1% [62.7, 67.8].
  Under the reversed-role corruption the opposite signal left the margin near baseline (−0.09 [−0.19, 0.01] and 0.04
  [−0.14, 0.25] logits), so there D measures the benefit alone.
- **Head 12.3 feeds the queries (post hoc, saved traces of 16 recipients, no intervals).** Holding A at its unedited
  value left 39–51% of the order-dependent change in the five queries; transplanting A alone recreated 57–69% of it;
  holding MLP 14 left 92–98%. Holding the queries left 98–101% of head 15.4's change. Heads 14.14 and 15.14, which only
  head 12.3 can reach, kept 29–45% and 40–52% of their change with A held.

## Why D differs between data sets (post hoc)

D depends on wording. It is largest in F10, the wording of the source sentences: 4.69 [3.97, 5.44] against 3.49
[2.78, 4.23] in F11 (Study 2 confirmation); Study 3's new wordings gave 3.26 [2.75, 3.77] (F12) and 2.64 [2.11, 3.27]
(F13). On the shared development data the two studies gave identical contrasts (4.31 in F10, 2.53 in F11). The lower D
of Study 3's confirmation set (2.95 against 4.09) therefore reflects its new wordings and names; fractions are
compared within a study.

## What it means

The fixed signal acts as a relative-position cue: it carries where the repeated name stood in the source sentence.
Where the recipient has the same name order, the name-mover heads use it as intended; where the order is reversed, it
points them at the wrong name. This order dependence had been shown qualitatively in GPT-2 Small: Wang et al.
(ICLR 2023) found that the S-inhibition signal carries the position of the repeated name, Merullo, Eickhoff and
Pavlick ("Talking Heads", NeurIPS 2024) found that a low-rank component of S-inhibition head 8.10 strengthens or
weakens inhibition according to which name comes first, and Kissane et al. (2024) found that the positional signal
records whether the repeated name is the one that follows "and". What this project adds is the measurement in GPT-2
Medium: one cached signal crossed with recipient name order, wording, names and places, effect sizes with intervals,
a pathway selected on development data and confirmed on frozen new data, and the shares of the effect that its
components carry. In post hoc analyses the identified components carry similar shares of the benefit and of the harm.
That is consistent with one computation reading the signal in both orders, and also with order-specific routes into
the same components, as Franco et al. (2026) found for the two name orders in GPT-2 Small. These components carry
about 70% of the effect in one setting, and 29.1% remains unexplained; the development screen points to further,
overlapping final-position components for the rest.

**Limits.** One model, one family of sentences, one cached signal, strengths up to 1; interventions at the final
position only; the two added heads were confirmed as a group; each study's hypotheses came from the previous study's
data, and the freezes are local records made minutes apart; Study 1's confirmation names are new to this project but
mostly occur in the prompts of the companion study that chose the twelve heads, so generalisation to unseen names
rests on Studies 2 and 3.
