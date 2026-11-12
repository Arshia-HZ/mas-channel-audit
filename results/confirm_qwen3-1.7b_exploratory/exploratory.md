EXPLORATORY: descriptive, written after the analysis freeze; not a preregistered result (PREREG 10.2, item 9)

Qwen/Qwen3-1.7B: 600 problems x 2 replicates. Means over problems; 95% percentile bootstrap intervals, unadjusted.

== change types (share of agents) ==
arm                wrong>right right>wrong wrong>other  stay right  stay wrong   net (right-to-wrong subtracted)
self_reflect              3.8%        0.6%        7.1%       63.3%       25.2%   +3.2% [+2.4%, +4.0%]
answer_only               3.8%        0.6%        6.7%       63.3%       25.6%   +3.2% [+2.4%, +4.0%]
answer_conf               3.9%        1.4%        6.8%       62.5%       25.3%   +2.5% [+1.6%, +3.4%]
dissent_only              3.4%        0.3%        5.4%       63.6%       27.3%   +3.1% [+2.4%, +3.8%]
failure_cause             4.2%        0.9%        8.4%       63.0%       23.4%   +3.3% [+2.4%, +4.2%]
full_solution            10.8%        4.2%       13.3%       59.7%       12.0%   +6.6% [+5.2%, +7.9%]
placebo                   2.6%       13.4%       17.4%       50.5%       16.1%   -10.8% [-12.4%, -9.2%]
failure_cause_v2          3.6%        0.7%        7.2%       63.2%       25.3%   +2.9% [+2.1%, +3.7%]

== source of change: spontaneous + form + content = total ==
-- wrong_to_correct --
arm                 spontaneous     form  content    total
answer_only                3.8%    -1.2%     1.2%     3.8%
answer_conf                3.8%    -1.2%     1.3%     3.9%
dissent_only               3.8%    -1.2%     0.8%     3.4%
failure_cause              3.8%    -1.2%     1.6%     4.2%
full_solution              3.8%    -1.2%     8.2%    10.8%
placebo                    3.8%    -1.2%     0.0%     2.6%
failure_cause_v2           3.8%    -1.2%     1.0%     3.6%
-- correct_to_wrong --
arm                 spontaneous     form  content    total
answer_only                0.6%    12.8%   -12.8%     0.6%
answer_conf                0.6%    12.8%   -12.0%     1.4%
dissent_only               0.6%    12.8%   -13.1%     0.3%
failure_cause              0.6%    12.8%   -12.5%     0.9%
full_solution              0.6%    12.8%    -9.2%     4.2%
placebo                    0.6%    12.8%     0.0%    13.4%
failure_cause_v2           0.6%    12.8%   -12.7%     0.7%
-- net --
arm                 spontaneous     form  content    total
answer_only                3.2%   -14.0%    14.0%     3.2%
answer_conf                3.2%   -14.0%    13.3%     2.5%
dissent_only               3.2%   -14.0%    13.9%     3.1%
failure_cause              3.2%   -14.0%    14.1%     3.3%
full_solution              3.2%   -14.0%    17.4%     6.6%
placebo                    3.2%   -14.0%     0.0%   -10.8%
failure_cause_v2           3.2%   -14.0%    13.7%     2.9%

== diversity (distinct parsable answers among the three agents) ==
round 0: 1.47 distinct, 61.4% with at most one
arm                  after    change  at most one
self_reflect          1.59     +0.12        56.7%
answer_only           1.57     +0.10        57.3%
answer_conf           1.59     +0.12        55.3%
dissent_only          1.53     +0.06        59.6%
failure_cause         1.59     +0.12        56.1%
full_solution         1.29     -0.18        75.8%
placebo               1.99     +0.52        34.0%
failure_cause_v2      1.60     +0.13        55.8%

== headroom: recovered = a wrong round-0 plurality with a right answer present (31 cases); retained = a right plurality (801 cases) ==
arm                                recovered                  retained
self_reflect             20.4% [7.4%, 36.1%]      99.3% [98.7%, 99.7%]
answer_only               7.5% [0.0%, 17.6%]      99.3% [98.8%, 99.7%]
answer_conf              18.3% [6.2%, 33.3%]      98.8% [98.0%, 99.5%]
dissent_only             10.8% [1.1%, 22.9%]      99.3% [98.7%, 99.7%]
failure_cause            23.7% [9.7%, 39.5%]      99.1% [98.5%, 99.6%]
full_solution           36.6% [20.4%, 53.8%]      97.2% [96.0%, 98.2%]
placebo                  19.4% [8.3%, 32.4%]      85.3% [82.9%, 87.6%]
failure_cause_v2         16.1% [3.8%, 30.6%]      99.2% [98.7%, 99.6%]

== conformity: moving to a peer's round-0 answer ==
arm                 exposed wrong        copied wrong | exposed       adopted right | exposed
self_reflect      33.5% [30.6%, 36.3%]             2.7% [1.8%, 3.7%]          19.1% [15.4%, 23.1%]
answer_only       33.5% [30.6%, 36.3%]             5.0% [3.7%, 6.4%]          19.3% [15.4%, 23.4%]
answer_conf       33.5% [30.6%, 36.3%]             3.5% [2.4%, 4.7%]          20.8% [17.0%, 24.8%]
dissent_only      33.5% [30.6%, 36.3%]             3.0% [2.0%, 4.1%]          18.4% [14.8%, 22.2%]
failure_cause     33.5% [30.6%, 36.3%]             3.3% [2.3%, 4.5%]          21.5% [17.6%, 25.6%]
full_solution     33.5% [30.6%, 36.3%]          33.4% [30.2%, 36.5%]          64.3% [59.7%, 69.1%]
placebo           33.5% [30.6%, 36.3%]             2.2% [1.4%, 3.1%]          13.7% [10.5%, 17.0%]
failure_cause_v2  33.5% [30.6%, 36.3%]             2.7% [1.8%, 3.8%]          18.2% [14.7%, 22.0%]

EXPLORATORY: descriptive, written after the analysis freeze; not a preregistered result (PREREG 10.2, item 9)
