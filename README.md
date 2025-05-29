# MT Exercise 4: Byte Pair Encoding, Beam Search

## Data Preprocessing

For this project, I used the English-to-German (en → de) translation direction and performed different preprocessing steps depending on whether the model used word-level or subword-level input.
The original training files were:
```
data/train.en-de.en  
data/train.en-de.de
```
I randomly selected 100,000 sentence pairs:
```
shuf -n 100000 data/train.en-de.en > data/train.100k.en
shuf -n 100000 data/train.en-de.de > data/train.100k.de
```
All data (train, dev, test) were tokenized using `sacremoses`：
```
sacremoses -l en tokenize < data/train.100k.en > data/train.tok.en
sacremoses -l de tokenize < data/train.100k.de > data/train.tok.de

sacremoses -l en tokenize < data/dev.en-de.en > data/dev.tok.en
sacremoses -l de tokenize < data/dev.en-de.de > data/dev.tok.de

sacremoses -l en tokenize < data/test.en-de.en > data/test.tok.en
sacremoses -l de tokenize < data/test.en-de.de > data/test.tok.de
```

- **Word-Level Model**

For the word-based model (word_model), I used moses tokenizer for both source (English) and target (German) sides.
The resulting files used were:
```
data/train.tok.en
data/train.tok.de
data/dev.tok.en
data/dev.tok.de
data/test.tok.en
data/test.tok.de
```
- **BPE Models**

1. We use tokenized English and German data as input:
```
data/train.tok.en / train.tok.de  
data/dev.tok.en / dev.tok.de  
data/test.tok.en / test.tok.de
```

2. We learn joint BPE codes with a target vocabulary size of 2000, 4000, 6000:
```
subword-nmt learn-joint-bpe-and-vocab \
  --input data/train.tok.en data/train.tok.de \
  -s 2000 \
  --total-symbols \
  -o data/bpe2000.codes \
  --write-vocabulary data/vocab.en data/vocab.de
```
I repeated this with 4000 and 6000 for the other experiments.

3. Applying BPE:
```
# English
subword-nmt apply-bpe -c data/bpe2000.codes < data/train.tok.en > data/train.bpe2000.en
subword-nmt apply-bpe -c data/bpe2000.codes < data/dev.tok.en   > data/dev.bpe2000.en
subword-nmt apply-bpe -c data/bpe2000.codes < data/test.tok.en  > data/test.bpe2000.en
# German
subword-nmt apply-bpe -c data/bpe2000.codes < data/train.tok.de > data/train.bpe2000.de
subword-nmt apply-bpe -c data/bpe2000.codes < data/dev.tok.de   > data/dev.bpe2000.de
subword-nmt apply-bpe -c data/bpe2000.codes < data/test.tok.de  > data/test.bpe2000.de
```
4. Create a joint vocabulary file for JoeyNMT
```      
cat data/vocab.en data/vocab.de | cut -f1 | sort | uniq > data/vocab.bpe2000
```
This was done separately for each vocab size (2000 / 4000 / 6000)，each BPE model has its own set of .codes, .vocab, and BPE-tokenized data files.
Example filenames:
```
data/train.bpe2000.en, data/train.bpe2000.de, ...
data/train.bpe4000.en, data/train.bpe4000.de, ...
```

## Files and Scripts

1. Based on transformer_sample_config.yaml, we created:
```
word_model.yaml: for word-level model, using voc_limit: 2000
bpe2000_model.yaml: for BPE model with vocab size 2000
bpe4000_model.yaml: for BPE model with vocab size 4000
bpe6000_model.yaml: for BPE model with vocab size 6000 
```
- For word-level model: 
  - level: word	
  - voc_limit: 2000	
  - voc_file: null	
  - tokenizer_type: space
- For BPE models:
  - level: bpe
  - voc_limit: null
  - voc_file: path to manually created vocab file (data/vocab.bpe2000)
  - tokenizer_type: subword-nmt
  - tokenizer_cfg.codes: path to BPE codes (data/bpe2000.codes)
2. We modified `train.sh` to accept the model name (model_name) as a command-line argument instead of hardcoding it. 
3. `evaluate.sh`for the word-level model:
- sets ext=tok to refer to tokenized files;
- uses the word_model config.
4. `evaluate_bpe.sh` for bpe_model ：
- accepts `$1` as the model name (model_name)，also accepts an optional `$2` parameter for the file extension (ext), defaulting to $1.
- uses data/ directory for preprocessed data.
- dynamically sets input test file path as `$data/test.$ext.$src`
- saves translation output under `$translations/$model_name/test.$model_name.$trg.`

## BLEU Scores Comparison

| Model Type      | Vocabulary Size | BLEU Score        |
|-----------------|-----------------|-------------------|
| Word-Level      | 2000            | 13.0              |
| BPE             | 2000            | 0.0               |
| BPE             | 4000            | 0.0               |
| BPE             | 6000            | 0.0               |

**Note:**  
- The word-level model achieves reasonable BLEU scores.  
- All BPE-based models produced BLEU scores of 0.0，the generated hypotheses consisted mostly of /unk/ tokens, indicating that the models failed to learn meaningful subword representations.
```
{
 "name": "BLEU",
 "score": 0.0,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1",
 "verbose_score": "0.0/0.0/0.0/0.0 (BP = 1.000 ratio = 3.752 hyp_len = 283098 ref_len = 75454)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.5.1"
}
```
- To address this, I tried switching to a different dataset (German-Italian) with BPE2000 preprocessing and training. In this case, the hypotheses were no longer dominated by /unk/, but the outputs were repetitive tokens like “di di di …” and generally nonsensical, showing that the model still failed to learn proper translations.


# Impact of Beam Size on Translation Quality

I chose to use the **word_model** because the other models I tried, such as the bpe2000_model, resulted in BLEU scores of zero, which made those models unusable for meaningful comparison. Therefore, the word-level model was the only option.

The script used to generate the graphs (`beam_graph.py`)and the resulting plot images are stored in the `beam_search` folder.

I used the same evaluation shell script (`evaluate.sh`), but made two key changes for each run:

1. I manually changed the `beam_size` parameter inside the `word_model.yaml`.
2. I updated the `beam_size` variable inside `evaluate.sh` accordingly, and modified the output file name to include the beam size, so that the translations are saved separately and do not overwrite each other.

From the results:

1. BLEU scores gradually improve as beam size increases, peaking at beam sizes 4 and 6 with a score of 13.1. After that, the score stays almost the same.
2. Translation time generally increases with larger beam sizes.
3. An unusual point is that beam size 1 took unusually longer than beam size 2, which might be due to some overhead or initialization effects.
4. Beam size 4 seems like the best choice because it achieves the highest BLEU score without taking too much extra time.
