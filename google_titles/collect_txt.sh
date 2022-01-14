#!/bin/bash


file_names=('BrowserHistoryAS.json'
            'BrowserHistory_bekendt.json'
            'BrowserHistory_BL.json'
            'BrowserHistoryKG.json'
            'BrowserHistory_KOE.json'
            'BrowserHistory_LLT.json' 
            'BrowserHistory_LP.json' 
            'BrowserHistoryMB.json'
            'BrowserHistory_MNN.json' 
            'BrowserHistoryVM.json');

            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 each_file_txt.py $my_file_name

done

