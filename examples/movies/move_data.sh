data=(1000419844110211017 1000660235447900098 100370370781588123 1003808605201549184 1004085618827838148)

for folder in "${data[@]}"; do
    ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /pscratch/sd/r/richardr/v5/20³ && tar cz $folder" | tar zxv -C /Users/richardr2926/Desktop/Research/Code/dfno/data/DFNO_3D/v5/20³
done
