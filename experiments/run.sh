#!/bin/bash

function func {
    name=$occlude'_'$color'/'$object
    path_data=$folder_data'/'$name'_2_3.h5'
    folder_log='logs/'$name
    folder_out=$name
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
    path_data=$folder_data'/'$name'_4.h5'
    for obj_slots in 4 10; do
        file_result='general_'$obj_slots'.h5'
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder_log $folder_log \
            --folder_out $folder_out \
            --num_slots $(( obj_slots + 1 )) \
            --file_result $file_result
    done
}

export CUDA_VISIBLE_DEVICES='0'
run_file='../src/main.py'
folder_data='../data'
path_config='config.yaml'

for occlude in 'sep' 'occ'; do
    for color in 'gray' 'rgb_1' 'rgb_2' 'rgb_3' 'rgb_4'; do
        for object in 'shapes' 'mnist'; do
            func
        done
    done
done
