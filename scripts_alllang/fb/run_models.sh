#!/bin/bash


file_names=('your_posts_1AF.txt'
           # 'your_posts_1_AG2.txt'
           # 'your_posts_1_AG.txt'
            'your_posts_1_AIH.txt'
            'your_posts_1_AMB.txt'
            'your_posts_1_ASA.txt' 
            'your_posts_1_AS.txt' 
            'your_posts_1_BE.txt'
            'your_posts_1_BL.txt' 
            'your_posts_1_EBJ.txt' 
           # 'your_posts_1_GN.txt'
           # 'your_posts_1_HHO.txt' 
            'your_posts_1_IMJH.txt');
           # 'your_posts_1_KB.txt'
           # 'your_posts_1_KG.txt'
           # 'your_posts_1_LJP.txt'
           # 'your_posts_1_LRE.txt'
           # 'your_posts_1MB.txt'
           # 'your_posts_1_SB.txt'
           # 'your_posts_1_SRP.txt' 
           # 'your_posts_1_TJJ.txt'
           # 'your_posts_1_TLHS.txt'
           # 'your_posts_1_UP.txt'
           # 'your_posts_1_VT.txt'
           # 'your_posts_2_VT.txt'
           # 'your_posts_VM.txt');
           
             
#'your_posts_CML.txt', 'your_posts_LF.txt' -> too small
            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 20_fb.py $my_file_name

done

