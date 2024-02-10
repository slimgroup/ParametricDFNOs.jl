# nodes gpus nblocks, dim, md, mt, ntrain, nvalid, nbatch, epochs

# Base model
bash examples/perlmutter/short_train.sh 4 16 4 20 4 8 1 1 1 200

# 6 sconv blocks
bash examples/perlmutter/short_train.sh 4 16 6 20 4 8 1 1 1 200

# twice the number of modes
bash examples/perlmutter/short_train.sh 4 16 4 20 8 16 1 1 1 200

# bash examples/perlmutter/short_train.sh 4 16 10 20 2 2
# bash examples/perlmutter/short_train.sh 25 100 10 40 2 2
# bash examples/perlmutter/short_train.sh 50 200 10 80 2 2
# bash examples/perlmutter/short_train.sh 100 400 10 160 2 2
