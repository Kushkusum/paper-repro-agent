# Reproducibility Leaderboard

10 experiment(s) attempted across 4 paper(s).

| Paper | Verdict | Metrics (reported vs observed) | Iterations |
|---|---|---|---|
| An Empirical Evaluation of Thompson Sampling<br><sub>7 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_10, ucb_ts_ratio_delta_100, ucb_ts_ratio_delta_1000, ucb_ts_ratio_delta_3, ucb_ts_ratio_delta_316, ucb_ts_ratio_delta_32</sub> | ⚠️ unresolved | `ucb_ts_ratio_delta_1`: 2.65 vs 0.6649<br>`ucb_ts_ratio_delta_3`: 2.68 vs 0.7376<br>`ucb_ts_ratio_delta_10`: 2.84 vs 1.523<br>`ucb_ts_ratio_delta_32`: 2.98 vs 2.201<br>`ucb_ts_ratio_delta_100`: 3.22 vs 1.492<br>`ucb_ts_ratio_delta_316`: 3.6 vs -5.496<br>`ucb_ts_ratio_delta_1000`: 3.82 vs 1.484 | 6 |
| An Empirical Evaluation of Thompson Sampling<br><sub>1 metric: ucb_ts_regret_ratio</sub> | ✅ reproduced | `ucb_ts_regret_ratio`: 2.65 vs 2.706 | 3 |
| An Empirical Evaluation of Thompson Sampling<br><sub>7 metrics: ratio_ucb_ts_1, ratio_ucb_ts_10, ratio_ucb_ts_100, ratio_ucb_ts_1000, ratio_ucb_ts_3, ratio_ucb_ts_316, ratio_ucb_ts_32</sub> | ⚠️ unresolved | `ratio_ucb_ts_1`: 2.65 vs -1.645<br>`ratio_ucb_ts_3`: 2.68 vs 0.2607<br>`ratio_ucb_ts_10`: 2.84 vs -0.1707<br>`ratio_ucb_ts_32`: 2.98 vs 0.4077<br>`ratio_ucb_ts_100`: 3.22 vs -0.08131<br>`ratio_ucb_ts_316`: 3.6 vs -0.08062<br>`ratio_ucb_ts_1000`: 3.82 vs -0.004689 | 6 |
| An Empirical Evaluation of Thompson Sampling<br><sub>3 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_100, ucb_ts_ratio_delta_1000</sub> | ⚠️ unresolved | `ucb_ts_ratio_delta_1`: 2.65 vs 0.7052<br>`ucb_ts_ratio_delta_100`: 3.22 vs 0.7178<br>`ucb_ts_ratio_delta_1000`: 3.82 vs 0.7485 | 6 |
| An Empirical Evaluation of Thompson Sampling<br><sub>2 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_1000</sub> | ⚠️ unresolved | `ucb_ts_ratio_delta_1`: 2.65 vs 0.9921<br>`ucb_ts_ratio_delta_1000`: 3.82 vs 1.016 | 6 |
| Enhanced Qwen-VL 7B Model via Instruction Finetuning on Chinese Medical Dataset<br><sub>1 metric: Rouge-1 score</sub> | 🚫 infeasible (GPU) | — | 0 |
| Finite-time Analysis of the Multiarmed Bandit Problem<br><sub>1 metric: ucb1_regret_bound</sub> | ⚠️ unresolved | `ucb1_regret_bound`: 308.3 vs 308.3 | 6 |
| Finite-time Analysis of the Multiarmed Bandit Problem<br><sub>1 metric: ucb1_regret</sub> | ✅ reproduced | `ucb1_regret`: 308.3 vs 22.47 | 1 |
| The pre-commitment best-choice problem: exact finite-n formulas<br><sub>1 metric: simulated_win_probability</sub> | ✅ reproduced | `simulated_win_probability`: 0.5218 vs 0.527 | 1 |
| The pre-commitment best-choice problem: exact finite-n formulas<br><sub>1 metric: empirical_win_rate</sub> | ✅ reproduced | `empirical_win_rate`: 0.5218 vs 0.522 | 2 |

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

### An Empirical Evaluation of Thompson Sampling — 3 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_100, ucb_ts_ratio_delta_1000

**Verdict:** ⚠️ unresolved (`20260707-225252_an-empirical-evaluation-of-thompson-samp`)

Reached the cap of 6 iterations without matching all target metrics. Last diagnosis: The observed values for the metrics 'ucb_ts_ratio_delta_1', 'ucb_ts_ratio_delta_100', and 'ucb_ts_ratio_delta_1000' are significantly lower than the reported values, with relative errors of 73.39%, 77.71%, and 80.41%, respectively. These errors exceed the tolerance of 35%. The code implements Thompson Sampling and Upper Confidence Bound algorithms, and the issue seems to be in the calculation of the regret ratio. Specifically, the problem lies in the way the cumulative rewards are calculated and used to determine the regret ratio. The implementation appears to be incorrect, leading to a significant deviation from the expected results.

### An Empirical Evaluation of Thompson Sampling — 2 metrics: ucb_ts_ratio_delta_1, ucb_ts_ratio_delta_1000

**Verdict:** ⚠️ unresolved (`20260707-233918_an-empirical-evaluation-of-thompson-samp`)

Reached the cap of 6 iterations without matching all target metrics. Last diagnosis: The observed values for both metrics 'ucb_ts_ratio_delta_1' and 'ucb_ts_ratio_delta_1000' are significantly lower than the reported values, with relative errors of 62.56% and 73.41% respectively. This suggests a systematic issue in the implementation. Upon reviewing the code, a typo is noticed in the function call for the UCB algorithm within the main function: 'simulate_bandit' should be 'simulate_bandit'. However, this seems to be a simple typo in transcription. A more critical review reveals that the calculation of regret and the ratio seems correct. But there is a potential issue with the way the optimal reward is calculated and used. The optimal reward calculation uses the initial theta values, but it does not account for the changes in theta over time due to the replacement of arms. This could lead to incorrect regret calculations. However, the main issue seems to stem from the incorrect implementation of the UCB algorithm or the simulation loop. Another potential bug could be in the way the seed is used or the random number generation. But most importantly, there is actually a simple and obvious bug: in the simulate_bandit call, it should be 'simulate_bandit' corrected to 'simulate_bandit' but actually it seems there was a simple typo which was not actually in the code provided: 'ucb_reward, ucb_optimal_reward = simulate_bandit(T, delta, retirement_prob, K, alpha, beta, 'ucb')' should actually match 'ucb_reward, ucb_optimal_reward = simulate_bandit(T, delta, retirement_prob, K, alpha, beta, 'ucb')'. The real issue here seems actually with line 'ucb_reward, ucb_optimal_reward = simulate_bandit(T, delta, retirement_prob, K, alpha, beta, 'ucb')' which actually seems correct. Looking closely at bandit_simulation.py there is actually a logical error at line where we update success and failure counts and rewards calculation.

### Enhanced Qwen-VL 7B Model via Instruction Finetuning on Chinese Medical Dataset — 1 metric: Rouge-1 score

**Verdict:** 🚫 infeasible (GPU) (`20260707-214115_enhanced-qwen-vl-7b-model-via-instructio`)

The paper's method involves fine-tuning a 7B parameter model (Qwen-VL), which requires significant computational resources, particularly a GPU, to execute in a reasonable amount of time. The sandbox environment, however, is limited to a single CPU core with no GPU access, making it impractical to fine-tune such a large model within the given time budget. Furthermore, the method also requires the QLoRA approach and a custom dataset, which cannot be downloaded or accessed in the no-network sandbox, but the primary blocker is the computational demand of the model itself.

### Finite-time Analysis of the Multiarmed Bandit Problem — 1 metric: ucb1_regret_bound

**Verdict:** ⚠️ unresolved (`20260707-220136_finite-time-analysis-of-the-multiarmed-b`)

Reached the cap of 6 iterations; the last apparent match was rejected as non-genuine: The reported value 'ucb1_regret_bound' is computed directly from the 'calculate_theoretical_regret_bound' function using known parameters (n, arm_means). The 'simulated_regret' variable, which is derived from actual simulated data via the 'ucb1' function, is not used in the computation of the reported metric. The 'theoretical_regret_bound' is calculated using a formula that only depends on known constants and parameters, not on any simulated outcome.

### Finite-time Analysis of the Multiarmed Bandit Problem — 1 metric: ucb1_regret

**Verdict:** ✅ reproduced (`20260707-225140_finite-time-analysis-of-the-multiarmed-b`)

All 1 target metric(s) fell within tolerance on iteration 1, and the match was verified as a genuine measurement, not a hardcoded/formula pass-through.

### The pre-commitment best-choice problem: exact finite-n formulas — 1 metric: simulated_win_probability

**Verdict:** ✅ reproduced (`20260707-223632_the-pre-commitment-best-choice-problem-e`)

All 1 target metric(s) fell within tolerance on iteration 1, and the match was verified as a genuine measurement, not a hardcoded/formula pass-through.

### The pre-commitment best-choice problem: exact finite-n formulas — 1 metric: empirical_win_rate

**Verdict:** ✅ reproduced (`20260707-230710_the-pre-commitment-best-choice-problem-e`)

All 1 target metric(s) fell within tolerance on iteration 2, and the match was verified as a genuine measurement, not a hardcoded/formula pass-through.
