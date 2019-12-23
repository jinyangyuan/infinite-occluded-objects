#!/bin/bash

function func_basic {
    mode_occlude=$1
    mode_color=$2
    mode_object=$3

    run_file='../src/main.py'
    folder_data='../data'
    path_config='config.yaml'
    folder=$mode_occlude'_'$mode_color'_'$mode_object

    path_data=$folder_data'/'$mode_occlude'_'$mode_color'/'$mode_object'_2_3_data.h5'
    for train in 1 0; do
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder $folder \
            --train $train
    done

    path_data=$folder_data'/'$mode_occlude'_'$mode_color'/'$mode_object'_4_data.h5'
    for max_objects in 4 10; do
        file_result_base='general_'$max_objects'_result_{}.h5'
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder $folder \
            --file_result_base $file_result_base \
            --train 0 \
            --max_objects $max_objects
    done
}

export CUDA_VISIBLE_DEVICES='0'

for mode_occlude in 'sep' 'occ'; do
    for mode_color in 'gray' 'rgb_1' 'rgb_2' 'rgb_3' 'rgb_4'; do
        for mode_object in 'shapes' 'mnist'; do
            func_basic $mode_occlude $mode_color $mode_object
        done
    done
done
