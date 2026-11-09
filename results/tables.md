# Results tables

## Table 1: Data

| Item | Value |
|---|---|
| Train / validation / test rows | 78799 / 8755 / 21892 |
| Min / max signal value (train file) | 0.0 / 1.0 |
| Rows per class, train file (N, S, V, F, Q) | 72471, 2223, 5788, 641, 6431 |
| Rows per class, test file (N, S, V, F, Q) | 18118, 556, 1448, 162, 1608 |

## Table 2: Network and training

| Item | Value |
|---|---|
| Trainable parameters | 32645 |
| Initial loss (log 5 = 1.6094) | 1.7174 |
| Gradient check max difference W[3] / b[3] | 0.00e+00 / 0.00e+00 |
| Chosen alpha | 0.5 |
| Best epoch (lowest validation loss) | 27 |
| Train / validation loss at best epoch | 0.0523 / 0.0740 |
| Train / validation accuracy at best epoch | 0.9834 / 0.9802 |
| Training time and hardware | 39.2 s on Tesla T4 |

## Table 3: Learning-rate choice

| alpha | Lowest validation loss | Validation accuracy at that epoch |
|---|---|---|
| 0.01 | 0.1618 | 0.9580 |
| 0.1 | 0.0877 | 0.9754 |
| 0.5 | 0.0740 | 0.9802 |

## Table 4: Test results

| | N | S | V | F | Q |
|---|---|---|---|---|---|
| Per-class accuracy | 0.9961 | 0.6439 | 0.9316 | 0.6543 | 0.9602 |

Test accuracy: 0.9777  
Balanced accuracy: 0.8372  
Always-N accuracy: 0.8276

### Test confusion matrix (rows = true class, columns = predicted)

| true \ pred | N | S | V | F | Q |
|---|---|---|---|---|---|
| N | 18047 | 20 | 39 | 7 | 5 |
| S | 186 | 358 | 11 | 1 | 0 |
| V | 68 | 2 | 1349 | 27 | 2 |
| F | 44 | 0 | 12 | 106 | 0 |
| Q | 45 | 3 | 16 | 0 | 1544 |
