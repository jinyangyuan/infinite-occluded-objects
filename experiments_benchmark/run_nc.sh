#!/bin/bash

function run_model {
    path_config='config_'$name'_nc.yaml'
    path_data=$folder_data'/'$name'.h5'
    folder_log='logs_nc/'$name
    folder_out='outs_nc/'$name
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
}

run_file='../src/main.py'
folder_data='../../compositional-scene-representation-datasets'

for name in 'mnist' 'dsprites' 'abstract' 'clevr' 'shop' 'gso'; do
    run_model
done
