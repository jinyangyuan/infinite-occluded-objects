#!/bin/bash

run_file='../src/main.py'
folder_data='../data'
gpu='0'

for mode_occlude in 'sep' 'occ'; do
    for mode_color in 'gray' 'rgb_1' 'rgb_2' 'rgb_3' 'rgb_4'; do
        if [ $mode_color == 'gray' ]; then image_channels=1; else image_channels=3; fi
        for mode_object in 'shapes' 'mnist'; do
            folder=$mode_occlude'_'$mode_color'_'$mode_object
            path_data=$folder_data'/'$mode_occlude'_'$mode_color'/'$mode_object'_2_3_data.h5'
            for train in 1 0; do
                python $run_file \
                    --gpu $gpu \
                    --path_data $path_data \
                    --folder $folder \
                    --train $train \
                    --image_channels $image_channels
            done
            path_data=$folder_data'/'$mode_occlude'_'$mode_color'/'$mode_object'_4_data.h5'
            train=0
            for max_objects in 4 10; do
                file_result_base='general_'$max_objects'_result_{}.h5'
                python $run_file \
                    --gpu $gpu \
                    --path_data $path_data \
                    --folder $folder \
                    --train $train \
                    --image_channels $image_channels \
                    --file_result_base $file_result_base \
                    --max_objects $max_objects
            done
        done
    done
done
