#!/bin/bash


file_names=('');
            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 main.py $my_file_name

done


