# har-behaviour-states: episodic retrieval for anticipating activity changes

**Can retrieving similar past moments anticipate what a person will do next, and is that real
anticipation or just the structure of the data-collection protocol?**

Tested on the UCI HAR smartphone dataset (30 people, 6 activities, 2.56 s windows with 50% overlap;
one step ahead = 1.28 s). Predictions are made 1.3, 2.6, 5.1 and 10.2 s ahead. The main metric is
**how many real activity changes a method anticipates**, since "the activity continues" is right most
of the time.

## Key findings

- **Retrieval anticipates some changes that persistence never does.** At 10.2 s ahead, population
  memory caught 32% of activity changes, against 2% for persistence and Markov models.
- **Most of its change alarms are wrong.** At best, 28% of its "change coming" predictions were
  correct, and its overall accuracy is lower than persistence (0.59 against 0.67 at 10.2 s).
- **The fixed collection protocol is a strong confound.** Every subject did the activities in the
  same order. A control that knows only how far into the session a person is caught 62% of changes
  at 0.70 precision, **but only when given the total session length**, which is not known at
  prediction time. Using the window index alone, it caught 43% at 0.26 precision by raising a change
  alarm in about half of all windows, with overall accuracy 0.51 against 0.59 for memory.
- **A person's own history is the weakest memory at every horizon** (14% of changes caught at
  10.2 s). Each person experienced each transition about once, so their own past rarely contains
  the change that is coming. Personal memory only helps when experiences repeat.

## Methods

| Method | What it uses |
|---|---|
| persistence | Classify the current window (logistic regression, 95.5% accuracy) and assume it continues |
| markov | Current predicted activity + h-step transition matrix from training subjects |
| script_position | **Control, no sensor data:** position in the session, either relative (t/n, uses the session length) or absolute (window index) |
| population_memory | k = 10 nearest past contexts from **other people** (training subjects), distance-weighted vote |
| population_memory_1perperson | As above, but the 10 neighbours come from 10 different people |
| own_history_memory | 10 nearest contexts from **the same person's earlier past** only |
| combined_memory | Nearest neighbours from both memories |

A context is the last 4 windows (30 PCA components each). Each prediction lists the retrieved past
moments (who, when, what happened next) as its explanation (`results/episodic_examples.txt`).

**Leakage guards:** test subjects never appear in population memory. Own-history memory only uses
windows whose future label was observed before the current context starts, which also excludes
overlapping windows. A data check confirms every subject's rows are contiguous and in time order
(11 to 16 activity changes per subject).

## Results at 10.2 s ahead (882 change moments, 9 test subjects)

| Method | Overall accuracy | Changes caught | Change alarms | Alarm precision |
|---|---|---|---|---|
| persistence | 0.668 | 1.7% | 0 | - |
| markov | 0.668 | 1.7% | 0 | - |
| script_position, knows session length | 0.831 | 61.7% | 771 | 0.696 |
| script_position, window index only | 0.508 | 42.7% | 1,465 | 0.255 |
| population_memory | 0.594 | 32.1% | 983 | 0.277 |
| population_memory_1perperson | 0.600 | 28.0% | 868 | 0.271 |
| own_history_memory | 0.337 | 13.9% | 1,412 | 0.082 |
| combined_memory | 0.570 | 25.1% | 922 | 0.229 |

Alarm precision = of the windows where a method predicts a change, the share where the activity
really changes and the method names the right new activity. All horizons are in
`results/episodic_results.csv` (relative control) and `results/abs_position/episodic_results.csv`
(window-index control).

## Limitations

- UCI HAR is scripted: activity changes follow the collection protocol, so this dataset cannot
  separate behaviour from protocol. Unscripted, in-the-wild data is needed for a real answer.
- Methods are compared at their own alarm rates, not at matched rates (no precision-recall curves).
- One dataset, one feature set (PCA of the provided features), k = 10 and context = 4 fixed.

## Reproduce

~~~bash
mkdir -p data && cd data
curl -L -o har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
unzip -q har.zip && unzip -q "UCI HAR Dataset.zip" && cd ..
pip install numpy scikit-learn
python3 episodic/episodic.py --data "data/UCI HAR Dataset"                  # relative-position control
python3 episodic/episodic.py --data "data/UCI HAR Dataset" --abs-position \
        --out results/abs_position                                          # window-index control
~~~
