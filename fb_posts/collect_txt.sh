#!/bin/bash


file_names=('your_posts_1AF.json'
            'your_posts_1_AG2.json'
            'your_posts_1_AG.json'
            'your_posts_1_AIH.json'
            'your_posts_1_AMB.json'
            'your_posts_1_ASA.json' 
            'your_posts_1_AS.json' 
            'your_posts_1_BE.json'
            'your_posts_1_BL.json' 
            'your_posts_1_EBJ.json' 
            'your_posts_1_GN.json'
            'your_posts_1_HHO.json' 
            'your_posts_1_IMJH.json'
            'your_posts_1_KB.json'
            'your_posts_1_KG.json'
            'your_posts_1_LJP.json'
            'your_posts_1_LRE.json'
            'your_posts_1MB.json'
            'your_posts_1_SB.json'
            'your_posts_1_SRP.json' 
            'your_posts_1_TJJ.json'
            'your_posts_1_TLHS.json'
            'your_posts_1_UP.json'
            'your_posts_1_VT.json'
            'your_posts_2_VT.json'
            'your_posts_CML.json'
            'your_posts_LF.json'
            'your_posts_VM.json');

            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 each_file_txt.py $my_file_name

done

