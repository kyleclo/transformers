### setup

Let's use `s2-server5` because it's always available.  It has super small GPUs.

This code is currently synced to Huggingface master `520e7f211926e07b2059bc8e21b668db4372e4db`.

### caching

```python
python create_cache_line_by_line_multiprocess.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --data_file=$TRAIN_FILE
```

### wikitext

First, let's see if we can get all the code working for wikitext.  Starting from the commands given in the huggingface README, we expand it to include all the hyperparameters.

```python
# for smaller dataset
export TRAIN_FILE=/net/nfs.corp/s2-research/kylel/scigpt_clean/data_third_1_2M/train.txt
export OUTPUT_DIR=/net/nfs.corp/s2-research/kylel/scigpt_clean/output_third_1_2M/
export PER_GPU_BATCH_SIZE=1
export NUM_GRAD_UPDATES=18750
export MODEL_NAME_OR_PATH=/net/nfs.corp/s2-research/kylel/scigpt_clean/output_first_2_4M/


# for larger datasets
export TRAIN_FILE=/net/nfs.corp/s2-research/kylel/scigpt_clean/data_second_2_0M/train.txt
export OUTPUT_DIR=/net/nfs.corp/s2-research/kylel/scigpt_clean/output_second_2_0M/
export PER_GPU_BATCH_SIZE=2
# export NUM_GRAD_UPDATES=37500
export NUM_GRAD_UPDATES=31250
export MODEL_NAME_OR_PATH=/net/nfs.corp/s2-research/kylel/scigpt_clean/output_first_2_4M/


# all of them:
export NUM_GPUS=4
export LR=1e-4
export TRUE_BATCH_SIZE=64
export WARMUP_PCT=0.05

export GRAD_ACCUM_STEPS=$((TRUE_BATCH_SIZE / NUM_GPUS / PER_GPU_BATCH_SIZE))
export CKPT_EVERY_X_UPDATES=$((NUM_GRAD_UPDATES / 10))

python -W ignore -m torch.distributed.launch --nproc_per_node $NUM_GPUS examples/run_language_modeling.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=gpt2 \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=$PER_GPU_BATCH_SIZE \
    --learning_rate=$LR \
    --max_steps=$NUM_GRAD_UPDATES \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --save_steps=$CKPT_EVERY_X_UPDATES \
    --warmup_pct=0.05 \
    --fp16 \
    --line_by_line \
    --cache_data_file_style=linebyline
    
    
    --cache_data_file_style=standard \
```

The training regime should be determined entirely by:

- number of gradient updates  (specified as `args.max_steps`)
- learning rate  (specified as `args.learning_rate`)
- true batch size  (specified as `args.true_batch_size`)
- warmup percentage  (specified as `args.warmup_pct`)

By default, Huggingface requires `args.gradient_accumulation_steps` and `args.warmup_steps` which can be derived from the latter 2 arguments above.

```python
gradient_accumulation_steps = true_batch_size // num_gpus // per_gpu_batch_size

warmup_steps = max_steps * warmup_pct
``` 

Other things:
`--line_by_line` is a nice idea, but it runs into errors with empty lines.  Need to modify, and not worth the time.


#### evaluation

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TEST_FILE=/net/nfs.corp/s2-research/kylel/scigpt_clean/data/val.txt

python eval_language_modeling.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --eval_data_file=$TEST_FILE \
    --per_gpu_eval_batch_size=$PER_GPU_BATCH_SIZE \
    --eval_all_checkpoints \
    --line_by_line \
    --cache_data_file_style=linebyline \
```