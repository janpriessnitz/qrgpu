#!/bin/bash


MATRIX_SIZES="1024 2048 4096 8192 16384 32768"

mkdir -p tuning/
cd qrgpu/
make tune
for rows in $MATRIX_SIZES; do
  for cols in $MATRIX_SIZES; do
    if (( cols > rows )); then
      continue
    fi
    echo "Tuning for matrix size ${rows}x${cols}"
    tuningDir="../tuning/${rows}/${cols}"
    mkdir -p $tuningDir
    echo "Created $tuningDir"
    cp * $tuningDir
    ./tune $rows $cols $tuningDir/TuningParams.h > $tuningDir/tuning.stdout

    pushd $tuningDir
    make main
    ./main ${rows} ${cols} > main.stdout
    popd
  done
done
