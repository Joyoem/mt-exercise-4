#!/bin/bash

model_name=word_model
base_dir=$(dirname "$0")/..
data=$base_dir/data
translations=$base_dir/translations/$model_name
mkdir -p $translations

ext=tok
src=en
trg=de
beam_size=$(grep 'beam_size:' $base_dir/configs/$model_name.yaml | awk '{print $2}')
output_file=$translations/test.beam${beam_size}.$trg

SECONDS=0
CUDA_VISIBLE_DEVICES=0 python -m joeynmt translate $base_dir/configs/$model_name.yaml < $data/test.$ext.$src > $output_file
time=$SECONDS

bleu=$(cat $output_file | sacrebleu $data/test.$ext.$trg --force | grep BLEU | cut -d " " -f 3 | cut -d "," -f 1)

echo "beam_size=$beam_size, BLEU=$bleu, time=${time}s"
echo "$beam_size,$bleu,$time" >> $translations/beam_bleu_time.csv