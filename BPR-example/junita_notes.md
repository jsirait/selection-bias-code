# Notes about the code for the Selection Bias paper

`gdelt-weekly-te` is the code to get the testing (or validation?) data (leave-one-out).
-----------------
`gdelt-weekly-te` is the code to get the training data.
-----------------

`gdelt` => I am not sure
-----------------

`gen_data` => I am not sure... to generate matrix `R`?
-----------------

`main` is the code for Matrix Factorization with AUC as metric rank and BPR as optimization method.

`main.m` has 900 lines of code, but code after line 177 are just for plotting, I think.

Details:
`M` events
`N` sources
`R_idx` is an nx2 matrix holding the indices of positive signals
`names` holds the string representation of sources
`[R_idx, M, N, names, ids] = gdelt(path, subset, reload);`

`Rtr` is the source-event interaction matrix -- is this our target matrix `R` as in the paper?

The low-rank matrices for sources `P` and events `Q` is randomly initialized (line 125-126 in `main.m`).

For checks and tests, they look at right-wing and left-wing sources (line 204 in `main.m`).

