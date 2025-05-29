#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data
configs=$base/configs
translations=$base/translations

mkdir -p $translations

src=en
trg=de

num_threads=4
device=0

# measure time

SECONDS=0
ext=tok
model_name=word_model

echo "###############################################################################"
echo "model_name $model_name"

beam_size=10  
translations_sub=$translations/$model_name
output_file=$translations_sub/test.beam$beam_size.$trg

mkdir -p $translations_sub

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt translate $configs/$model_name.yaml < $data/test.$ext.$src > $output_file

# compute case-sensitive BLEU 
cat $output_file | sacrebleu $data/test.$ext.$trg
echo "time taken:"
echo "$SECONDS seconds"