# Reproducibility Leaderboard

6 experiment(s) attempted across 4 paper(s).

| Paper | Verdict | Metrics (reported vs observed) | Iterations |
|---|---|---|---|
| An Empirical Evaluation of Thompson Sampling<br><sub>7 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_10, ucb_ts_ratio_delta_100, ucb_ts_ratio_delta_1000, ucb_ts_ratio_delta_3, ucb_ts_ratio_delta_316, ucb_ts_ratio_delta_32</sub> | ⚠️ unresolved | `ucb_ts_ratio_delta_1`: 2.65 vs 0.6649<br>`ucb_ts_ratio_delta_3`: 2.68 vs 0.7376<br>`ucb_ts_ratio_delta_10`: 2.84 vs 1.523<br>`ucb_ts_ratio_delta_32`: 2.98 vs 2.201<br>`ucb_ts_ratio_delta_100`: 3.22 vs 1.492<br>`ucb_ts_ratio_delta_316`: 3.6 vs -5.496<br>`ucb_ts_ratio_delta_1000`: 3.82 vs 1.484 | 6 |
| An Empirical Evaluation of Thompson Sampling<br><sub>1 metric: ucb_ts_regret_ratio</sub> | ✅ reproduced | `ucb_ts_regret_ratio`: 2.65 vs 2.706 | 3 |
| An Empirical Evaluation of Thompson Sampling<br><sub>7 metrics: ratio_ucb_ts_1, ratio_ucb_ts_10, ratio_ucb_ts_100, ratio_ucb_ts_1000, ratio_ucb_ts_3, ratio_ucb_ts_316, ratio_ucb_ts_32</sub> | ⚠️ unresolved | `ratio_ucb_ts_1`: 2.65 vs -1.645<br>`ratio_ucb_ts_3`: 2.68 vs 0.2607<br>`ratio_ucb_ts_10`: 2.84 vs -0.1707<br>`ratio_ucb_ts_32`: 2.98 vs 0.4077<br>`ratio_ucb_ts_100`: 3.22 vs -0.08131<br>`ratio_ucb_ts_316`: 3.6 vs -0.08062<br>`ratio_ucb_ts_1000`: 3.82 vs -0.004689 | 6 |
| Enhanced Qwen-VL 7B Model via Instruction Finetuning on Chinese Medical Dataset<br><sub>1 metric: Rouge-1 score</sub> | 🚫 infeasible (GPU) | — | 0 |
| Finite-time Analysis of the Multiarmed Bandit Problem<br><sub>1 metric: ucb1_regret_bound</sub> | ⚠️ unresolved | `ucb1_regret_bound`: 308.3 vs 308.3 | 6 |
| The pre-commitment best-choice problem: exact finite-n formulas<br><sub>1 metric: simulated_win_probability</sub> | ✅ reproduced | `simulated_win_probability`: 0.5218 vs 0.527 | 1 |

## Details

### An Empirical Evaluation of Thompson Sampling — 7 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_10, ucb_ts_ratio_delta_100, ucb_ts_ratio_delta_1000, ucb_ts_ratio_delta_3, ucb_ts_ratio_delta_316, ucb_ts_ratio_delta_32

**Verdict:** ⚠️ unresolved (`20260707-132944_an-empirical-evaluation-of-thompson-samp`)

Reached the cap of 6 iterations without matching all target metrics. Last diagnosis: The observed values for the metrics are consistently lower than the reported values, and in some cases, the ratio is negative or incorrect. This suggests a systematic issue with the implementation of the UCB and TS algorithms. Specifically, the large relative error and incorrect trend for certain delays (e.g., delta_316) indicate a potential bug in the code. The fact that the exit code is 0 and there is no error message in stderr suggests that the code did not crash, but the results are incorrect.

### An Empirical Evaluation of Thompson Sampling — 1 metric: ucb_ts_regret_ratio

**Verdict:** ✅ reproduced (`20260707-134119_an-empirical-evaluation-of-thompson-samp`)

All 1 target metric(s) fell within tolerance on iteration 3.

### An Empirical Evaluation of Thompson Sampling — 7 metrics: ratio_ucb_ts_1, ratio_ucb_ts_10, ratio_ucb_ts_100, ratio_ucb_ts_1000, ratio_ucb_ts_3, ratio_ucb_ts_316, ratio_ucb_ts_32

**Verdict:** ⚠️ unresolved (`20260707-142106_an-empirical-evaluation-of-thompson-samp`)

Reached the cap of 6 iterations without matching all target metrics. Last diagnosis: The observed values for the ratios of UCB regret to TS regret across different delays are consistently negative and deviate significantly from the reported values. This suggests a systematic issue in the implementation of either the Thompson Sampling or UCB algorithm. The negative ratios indicate that the regret calculated for UCB is often less than that of TS, or the regret for TS is negative, which does not make sense in the context of the problem. The issue likely lies in how regret is calculated in the functions `thompson_sampling` and `upper_confidence_bound`.

### Enhanced Qwen-VL 7B Model via Instruction Finetuning on Chinese Medical Dataset — 1 metric: Rouge-1 score

**Verdict:** 🚫 infeasible (GPU) (`20260707-214115_enhanced-qwen-vl-7b-model-via-instructio`)

The paper's method involves fine-tuning a 7B parameter model (Qwen-VL), which requires significant computational resources, particularly a GPU, to execute in a reasonable amount of time. The sandbox environment, however, is limited to a single CPU core with no GPU access, making it impractical to fine-tune such a large model within the given time budget. Furthermore, the method also requires the QLoRA approach and a custom dataset, which cannot be downloaded or accessed in the no-network sandbox, but the primary blocker is the computational demand of the model itself.

### Finite-time Analysis of the Multiarmed Bandit Problem — 1 metric: ucb1_regret_bound

**Verdict:** ⚠️ unresolved (`20260707-220136_finite-time-analysis-of-the-multiarmed-b`)

Reached the cap of 6 iterations; the last apparent match was rejected as non-genuine: The reported value 'ucb1_regret_bound' is computed directly from the 'calculate_theoretical_regret_bound' function using known parameters (n, arm_means). The 'simulated_regret' variable, which is derived from actual simulated data via the 'ucb1' function, is not used in the computation of the reported metric. The 'theoretical_regret_bound' is calculated using a formula that only depends on known constants and parameters, not on any simulated outcome.

### The pre-commitment best-choice problem: exact finite-n formulas — 1 metric: simulated_win_probability

**Verdict:** ✅ reproduced (`20260707-223632_the-pre-commitment-best-choice-problem-e`)

All 1 target metric(s) fell within tolerance on iteration 1, and the match was verified as a genuine measurement, not a hardcoded/formula pass-through.
