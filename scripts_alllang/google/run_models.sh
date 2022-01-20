#!/bin/bash


file_names=(#'BrowserHistoryAS.txt'
            #'BrowserHistory_bekendt.txt'
            'BrowserHistory_BL.txt'
            'BrowserHistoryKG.txt'
            'BrowserHistory_KOE.txt'
            'BrowserHistory_LLT.txt');
           # 'BrowserHistory_LP.txt' 
          #  'BrowserHistoryMB.txt'
         #   'BrowserHistory_MNN.txt'
         #   'BrowserHistoryVM.txt');
            
            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 20_g.py $my_file_name

done

