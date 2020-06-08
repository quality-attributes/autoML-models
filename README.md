# Iterated Racing Configuration for Differential Evolutionary Algorithms

Hyperparameter tuning for software requirements categorization using Differential Evolution

## Evolutionary Algorithms Configuration Results

### DecisionTreeClassifier

|       | Instance | Alive | Best |    Mean best | Exp so far |   W time |   rho | KenW |   Qvar |
| :---: | -------: | ----: | ---: | -----------: | ---------: | -------: | ----: | ---: | -----: |
|   x   |        1 |     8 |    2 | -58.12429379 |          8 | 00:07:39 |    NA |   NA |     NA |
|   x   |        2 |     8 |    2 | -58.28813559 |         16 | 00:07:07 | +0.81 | 0.90 | 0.2449 |
|   x   |        3 |     8 |    2 | -58.22598870 |         24 | 00:06:59 | +0.76 | 0.84 | 0.1995 |
|   x   |        4 |     8 |    2 | -58.02824859 |         32 | 00:07:03 | +0.80 | 0.85 | 0.1602 |
|   -   |        5 |     2 |    2 | -58.04180791 |         40 | 00:07:01 | +0.20 | 0.36 | 0.4000 |

Best-so-far configuration: 2
mean value: -58.04180791

Description of the best-so-far configuration:
|       | .ID. | population | generations | crossover | factor | .PARENT. |
| :---: | ---: | ---------: | ----------: | --------: | -----: | -------: |
|   2   |    2 |         28 |          97 |    0.6056 | 1.0345 |       NA |

Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):

|       | population | generations | crossover | factor |
| :---: | ---------: | ----------: | --------: | -----: |
|   2   |         28 |          97 |    0.6056 | 1.0345 |
|   7   |         34 |          78 |    0.1657 | 0.7599 |

### MultinomialNB

|       | Instance | Alive | Best |    Mean best | Exp so far |   W time |   rho | KenW |   Qvar |
| :---: | -------: | ----: | ---: | -----------: | ---------: | -------: | ----: | ---: | -----: |
|   x   |        1 |     8 |    1 | -63.18644068 |          8 | 00:00:51 |    NA |   NA |     NA |
|   x   |        2 |     8 |    1 | -63.35593220 |         16 | 00:00:49 | +0.39 | 0.70 | 0.6552 |
|   x   |        3 |     8 |    1 | -63.30131827 |         24 | 00:00:52 | +0.08 | 0.38 | 0.7576 |
|   x   |        4 |     8 |    1 | -63.27683616 |         32 | 00:00:49 | +0.14 | 0.36 | 0.7402 |
|   =   |        5 |     8 |    1 | -62.92203390 |         40 | 00:00:43 | +0.04 | 0.23 | 0.8468 |
|   =   |        6 |     8 |    1 | -62.74293785 |         48 | 00:00:49 | +0.05 | 0.21 | 0.8784 |

Best-so-far configuration: 1
mean value: -62.74293785

Description of the best-so-far configuration:

|       | .ID. | population | generations | crossover | factor | .PARENT. |
| :---: | ---: | ---------: | ----------: | --------: | -----: | -------: |
|   1   |    1 |         50 |          48 |    0.0778 | 1.1948 |       NA |

Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):

|       | population | generations | crossover | factor |
| :---: | ---------: | ----------: | --------: | -----: |
|   1   |         50 |          48 |    0.0778 | 1.1948 |
|   7   |         32 |          88 |    0.0998 | 1.6417 |
|   3   |         18 |          18 |    0.1956 | 1.5302 |
|   2   |         12 |          11 |    0.3356 | 1.6721 |

### SVC

|       | Instance | Alive | Best |    Mean best | Exp so far |   W time |   rho | KenW |   Qvar |
| :---: | -------: | ----: | ---: | -----------: | ---------: | -------: | ----: | ---: | -----: |
|   x   |        1 |     5 |    3 | -67.23163842 |          5 | 00:45:25 |    NA |   NA |     NA |
|   x   |        2 |     5 |    5 | -66.38983051 |         10 | 00:48:50 | +0.41 | 0.71 | 0.2080 |
|   x   |        3 |     5 |    3 | -66.78154426 |         15 | 00:48:01 | +0.34 | 0.56 | 0.2418 |
|   x   |        4 |     5 |    3 | -66.64124294 |         20 | 00:48:52 | +0.28 | 0.46 | 0.2893 |
|   =   |        5 |     5 |    3 | -66.55593220 |         25 | 00:46:40 | +0.28 | 0.42 | 0.2687 |
|   -   |        6 |     4 |    4 | -66.16666667 |         30 | 00:44:43 | -0.19 | 0.01 | 0.8539 |

Best-so-far configuration: 2
mean value: -66.05178908

Description of the best-so-far configuration:

|       | .ID. | population | generations | crossover | factor | .PARENT. |
| :---: | ---: | ---------: | ----------: | --------: | -----: | -------: |
|   2   |    2 |          6 |          75 |    0.5504 | 0.5324 |       NA |

Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
|       | population | generations | crossover | factor |
| :---: | ---------: | ----------: | --------: | -----: |
|   2   |          6 |          75 |    0.5504 | 0.5324 |
|   4   |          4 |          81 |    0.2506 | 1.2559 |
|   3   |         10 |          41 |    0.6035 | 1.3334 |
|   5   |         26 |          18 |    0.9067 | 0.3387 |

### RandomForestClassifier

|       | Instance | Alive | Best |    Mean best | Exp so far |   W time |   rho | KenW |   Qvar |
| :---: | -------: | ----: | ---: | -----------: | ---------: | -------: | ----: | ---: | -----: |
|   x   |        1 |     5 |    3 | -67.23163842 |          5 | 00:45:25 |    NA |   NA |     NA |
|   x   |        2 |     5 |    5 | -66.38983051 |         10 | 00:48:50 | +0.41 | 0.71 | 0.2080 |
|   x   |        3 |     5 |    3 | -66.78154426 |         15 | 00:48:01 | +0.34 | 0.56 | 0.2418 |
|   x   |        4 |     5 |    3 | -66.64124294 |         20 | 00:48:52 | +0.28 | 0.46 | 0.2893 |
|   =   |        5 |     5 |    3 | -66.55593220 |         25 | 00:46:40 | +0.28 | 0.42 | 0.2687 |
|   -   |        6 |     4 |    4 | -66.16666667 |         30 | 00:44:43 | -0.19 | 0.01 | 0.8539 |

Best-so-far configuration: 2
mean value: -66.05178908

Description of the best-so-far configuration:

|       | .ID. | population | generations | crossover | factor | .PARENT. |
| :---: | ---: | ---------: | ----------: | --------: | -----: | -------: |
|   2   |    2 |          6 |          75 |    0.5504 | 0.5324 |       NA |


Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):

|       | population | generations | crossover | factor |
| :---: | ---------: | ----------: | --------: | -----: |
|   2   |          6 |          75 |    0.5504 | 0.5324 |
|   4   |          4 |          81 |    0.2506 | 1.2559 |
|   3   |         10 |          41 |    0.6035 | 1.3334 |
|   5   |         26 |          18 |    0.9067 | 0.3387 |
