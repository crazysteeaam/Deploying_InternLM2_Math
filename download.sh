#!/bin/sh
# git clone git@hf.co:lmdeploy/turbomind-internlm-chat-20b-w4
# if [ ! -d "internlm2-chat-20b-4bits" ]
# then
#     echo "Downloading..."
#     git lfs clone https://huggingface.co/internlm/internlm2-chat-20b-4bits
# fi
# ls internlm/internlm2-chat-20b-4bits

if [ ! -d "internlm2-math-7b" ]
then
    echo "Downloading..."
    git lfs clone https://huggingface.co/internlm/internlm2-math-7b
fi
ls internlm/internlm2-math-7b