#!/bin/bash


file_names=('BrowserHistoryAS.pkl'
            'BrowserHistory_bekendt.pkl'
            'BrowserHistory_BL.pkl'
            'BrowserHistoryKG.pkl'
            'BrowserHistory_KOE.pkl'
            'BrowserHistory_LLT.pkl'
            'BrowserHistory_LP.pkl' 
            'BrowserHistoryMB.pkl'
            'BrowserHistory_MNN.pkl'
            'BrowserHistoryVM.pkl');
            
            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 model_each_file.py $my_file_name

done

