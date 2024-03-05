data=(1000419844110211017 1000660235447900098 100370370781588123 1003808605201549184 1004085618827838148)
v_data=(2967217681070558393 2967485641828085402 2970280804760794582 2970747949944738930 2973775305794135398)

for folder in "${v_data[@]}"; do
    ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /pscratch/sd/r/richardr/v5/20³ && tar cz $folder" | tar zxv -C /Users/richardr2926/Desktop/Research/Code/dfno/data/DFNO_3D/v5/20³
done
