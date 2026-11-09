Qwen/Qwen3-1.7B: 600 problems x 2 replicates

600 problems x 2 replicates

arm                     n   trunc  overflow  no box  no answer
answer_conf          3600    1.8%         0    0.1%       1.8%
answer_only          3600    1.6%         0    0.1%       1.6%
dissent_only         1703    1.8%         0    0.1%       1.8%
failure_cause        3600    1.3%         0    0.1%       1.2%
failure_cause_v2     3600    1.8%         0    0.1%       1.7%
failure_msg          4831    0.1%         0       -          -
failure_msg_v2       3770    0.6%         0       -          -
full_solution        3600    0.6%         0    0.0%       0.6%
placebo              3600    0.8%         0    0.1%       0.8%
round0               3600    8.2%         0    1.1%       9.0%
sc_pool              7200    7.6%         0    1.1%       8.4%
self_reflect         3600    1.5%         0    0.0%       1.4%

failure-message leakage (rejected = not clean on the first try)
  v1: 3600 messages, 2369 clean, 492 regenerated, 739 redacted (rejection 34.2%)
  v2: 3600 messages, 3430 clean, 70 regenerated, 100 redacted (rejection 4.7%)

G1 round-0 disagreement 38.6% (need 15%); by level: L3 29%, L4 34%, L5 53%
G2 recoverable headroom 2.6% (need 5%); by level: L3 2.0%, L4 3.4%, L5 2.4%
G3 pooled: overflow 0, failing arms [], flagged arms [], highest revision-arm truncation 1.8% -> PASS

SC accuracy by samples: k=1 63.7%, k=2 65.4%, k=3 69.7%, k=4 71.7%, k=5 73.4%, k=6 74.3%, k=7 75.2%, k=8 75.9%, k=9 76.8%

== main (PRIMARY; 600 problems; budget in generated tokens; Holm over contrasts 3 to 6) ==
arm                   acc  tokens   SC k  SC acc    gain
self_reflect        71.7%    4252   5.19   73.5%   -1.8%
answer_only         71.6%    4271   5.21   73.6%   -2.0%
answer_conf         72.0%    4269   5.21   73.6%   -1.6%
dissent_only        71.3%    3493   4.26   72.2%   -0.8%
failure_cause       72.0%    5799   7.07   75.3%   -3.3%
full_solution       71.8%    4209   5.14   73.5%   -1.7%
placebo             60.2%    4165   5.08   73.4%  -13.2%
failure_cause_v2    71.9%    4932   6.02   74.3%   -2.4%

contrast                                        mean             95% CI       p  p Holm  decision
full_solution_vs_sc                            -1.7%     [-3.3%, -0.0%]       -       -  equivalence not shown
answer_only_vs_full_solution                   -0.2%     [-1.6%, +1.2%]       -       -  non-inferiority not shown
placebo_vs_self_reflect_correct_to_wrong      +12.8%   [+11.4%, +14.3%]  0.0001  0.0004  rejected (Holm)
failure_cause_vs_full_solution                 -1.6%     [-3.1%, -0.0%]  0.0454  0.1362  not rejected (Holm)
answer_conf_vs_full_solution                   +0.1%     [-1.2%, +1.4%]  0.8917  0.8917  not rejected (Holm)
best_low_bandwidth_vs_sc                       -0.8%     [-2.0%, +0.3%]  0.1584  0.3168  not rejected (Holm)
failure_cause_v2_vs_full_solution              -0.7%     [-2.2%, +0.7%]  0.3208       -  secondary, unadjusted

== level_3 (secondary; 199 problems; budget in generated tokens; Holm over contrasts 3 to 6) ==
contrast                                        mean             95% CI       p  p Holm  decision
full_solution_vs_sc                            -2.9%     [-5.5%, -0.4%]       -       -  equivalence not shown
answer_only_vs_full_solution                   +0.6%     [-1.5%, +2.8%]       -       -  non-inferior
placebo_vs_self_reflect_correct_to_wrong      +13.2%   [+10.8%, +15.8%]  0.0001  0.0004  rejected (Holm)
failure_cause_vs_full_solution                 +0.7%     [-1.7%, +3.3%]  0.6015  1.0000  not rejected (Holm)
answer_conf_vs_full_solution                   +0.2%     [-1.9%, +2.2%]  0.9023  1.0000  not rejected (Holm)
best_low_bandwidth_vs_sc                       -1.1%     [-2.8%, +0.5%]  0.1840  0.5519  not rejected (Holm)
failure_cause_v2_vs_full_solution              +0.4%     [-1.8%, +2.7%]  0.7545       -  secondary, unadjusted

== level_4 (secondary; 193 problems; budget in generated tokens; Holm over contrasts 3 to 6) ==
contrast                                        mean             95% CI       p  p Holm  decision
full_solution_vs_sc                            -1.6%     [-4.2%, +0.8%]       -       -  equivalence not shown
answer_only_vs_full_solution                   +0.2%     [-2.2%, +2.5%]       -       -  non-inferiority not shown
placebo_vs_self_reflect_correct_to_wrong      +13.0%   [+10.3%, +15.7%]  0.0001  0.0004  rejected (Holm)
failure_cause_vs_full_solution                 -1.9%     [-4.4%, +0.5%]  0.1218  0.3654  not rejected (Holm)
answer_conf_vs_full_solution                   -0.1%     [-2.3%, +2.2%]  0.9485  1.0000  not rejected (Holm)
best_low_bandwidth_vs_sc                       -0.7%     [-2.6%, +1.2%]  0.5039  1.0000  not rejected (Holm)
failure_cause_v2_vs_full_solution              +0.4%     [-1.7%, +2.6%]  0.7275       -  secondary, unadjusted

== level_5 (secondary; 208 problems; budget in generated tokens; Holm over contrasts 3 to 6) ==
contrast                                        mean             95% CI       p  p Holm  decision
full_solution_vs_sc                            -0.5%     [-3.7%, +2.8%]       -       -  equivalence not shown
answer_only_vs_full_solution                   -1.4%     [-4.2%, +1.4%]       -       -  non-inferiority not shown
placebo_vs_self_reflect_correct_to_wrong      +12.3%   [+10.1%, +14.5%]  0.0001  0.0004  rejected (Holm)
failure_cause_vs_full_solution                 -3.3%     [-6.3%, -0.4%]  0.0244  0.0732  not rejected (Holm)
answer_conf_vs_full_solution                   +0.1%     [-2.5%, +2.7%]  0.9031  0.9871  not rejected (Holm)
best_low_bandwidth_vs_sc                       -0.9%     [-3.3%, +1.6%]  0.4936  0.9871  not rejected (Holm)
failure_cause_v2_vs_full_solution              -2.7%     [-5.6%, +0.0%]  0.0514       -  secondary, unadjusted

== no_round0_truncation (secondary; 482 problems; budget in generated tokens; Holm over contrasts 3 to 6) ==
contrast                                        mean             95% CI       p  p Holm  decision
full_solution_vs_sc                            -2.1%     [-3.8%, -0.4%]       -       -  equivalence not shown
answer_only_vs_full_solution                   -0.1%     [-1.7%, +1.4%]       -       -  non-inferiority not shown
placebo_vs_self_reflect_correct_to_wrong      +14.0%   [+12.3%, +15.8%]  0.0001  0.0004  rejected (Holm)
failure_cause_vs_full_solution                 -1.1%     [-2.8%, +0.5%]  0.1752  0.5255  not rejected (Holm)
answer_conf_vs_full_solution                   +0.0%     [-1.4%, +1.5%]  0.9551  0.9551  not rejected (Holm)
best_low_bandwidth_vs_sc                       -0.7%     [-1.8%, +0.5%]  0.2712  0.5423  not rejected (Holm)
failure_cause_v2_vs_full_solution              -0.4%     [-1.9%, +1.1%]  0.5767       -  secondary, unadjusted

== prompt_tokens_in_budget (secondary; 600 problems; budget in generated_plus_prompt tokens; Holm over contrasts 3 to 6) ==
contrast                                        mean             95% CI       p  p Holm  decision
full_solution_vs_sc                                                                      not estimable
answer_only_vs_full_solution                   -0.2%     [-1.6%, +1.2%]       -       -  non-inferiority not shown
placebo_vs_self_reflect_correct_to_wrong      +12.8%   [+11.4%, +14.3%]  0.0001  0.0002  rejected (Holm)
failure_cause_vs_full_solution                                                           not estimable
answer_conf_vs_full_solution                                                             not estimable
best_low_bandwidth_vs_sc                       -2.8%     [-4.1%, -1.6%]  0.0002  0.0002  rejected (Holm)
failure_cause_v2_vs_full_solution                                                        not estimable

not triggered: no_flagged_arm_truncation

Anything not listed above is exploratory and must be labelled so.
