#!/bin/bash

# Gray
folder_downloads='downloads'
folder_outputs_occ='occ_gray'
folder_outputs_sep='sep_gray'
folder_src='src'
for folder in $folder_downloads $folder_outputs_occ $folder_outputs_sep; do
    if [ ! -d $folder ]; then mkdir $folder; fi
done
for num_objects_all in '2 3' '4'; do
    if [[ $num_objects_all == '2 3' ]]; then
        num_train=50000
        num_valid=10000
        num_test=10000
    else
        num_train=0
        num_valid=0
        num_test=10000
    fi
    for occlusion in 0 1; do
        if [ $occlusion == 0 ]; then
            folder_outputs=$folder_outputs_sep
        else
            folder_outputs=$folder_outputs_occ
        fi
        python $folder_src/create_mnist.py \
            --folder_downloads $folder_downloads \
            --folder_outputs $folder_outputs \
            --occlusion $occlusion \
            --num_objects_all $num_objects_all \
            --num_train $num_train \
            --num_valid $num_valid \
            --num_test $num_test
        python $folder_src/create_shapes.py \
            --folder_outputs $folder_outputs \
            --occlusion $occlusion \
            --num_objects_all $num_objects_all \
            --num_train $num_train \
            --num_valid $num_valid \
            --num_test $num_test
    done
done

# RGB
for data_type in 'occ' 'sep'; do
    folder_inputs=$data_type'_gray'
    for only_object in 1 0; do
        for dependence in 0 1; do
            data_id=$(( (1 - only_object) * 2 + dependence + 1 ))
            folder_outputs=$data_type'_rgb_'$data_id
            if [ ! -d $folder_outputs ]; then mkdir $folder_outputs; fi
            for name in 'mnist' 'shapes'; do
                for num_objects_all in '2 3' '4'; do
                    python $folder_src'/convert_gray_to_rgb.py' \
                        --folder_inputs $folder_inputs \
                        --folder_outputs $folder_outputs \
                        --name $name \
                        --dependence $dependence \
                        --only_object $only_object \
                        --num_objects_all $num_objects_all
                done
            done
        done
    done
done
