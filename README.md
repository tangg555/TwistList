# TwistList: Resources and Baselines For Tongue Twister Generation
This repository is the code and resources for the paper [TwistList: Resources and Baselines For Tongue Twister Generation]() 

## Instructions

This project is mainly implemented with following tools:
- **Pytorch** 
- [pytorch-lightning](https://www.pytorchlightning.ai/) framework
- The initial checkpoints of pretrained models come from [Hugginface](https://huggingface.co).

So if you want to run this code, you must have following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/) (mine is 1.11.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) (mine is 4.21.3)
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) (mine is 1.8.2)

## Datasets and Resources

### Directly Download Dataset and Resources
To reproduce our work you need to download following files:

- Processed data (unzip them to be `datasets/tongue_twister` directory) [tongue_twister](https://www.dropbox.com/s/wfk9iibakcdb3jc/datasets.zip?dl=0)


- Raw data (put it to `resources/tongue_twister` directory) [raw data](https://www.dropbox.com/s/n170bxd2y3eevam/raw-data.zip?dl=0)

You need to download the raw data only if you want to reproduce the dataset by yourself.

### Preprocess Dataset From Scratch

Make sure you have `resources/tongue_twister/raw-data` ready.

Run `tasks/tongue_twister/enhance_dataset.py` firstly, and then `preprocess.py`.

### The introduction of the dataset
The structure of `datasets`should be like this:
```markdown
├── datasets/tongue_twister
   └── tt-data		# original tongue twister dataset
          └── `train.source.txt`    # input: key words 
          └── `train.target.txt`    # output: tongue twisters   
          └── `val.source.txt` 
          └── `val.target.txt` 
          └── `test.source.txt` 
          └── `test.target.txt` 
    └── tt-prompt-data		# we add a prompt for each input key words for tongue twister generation
          └── `train.source.txt`    
          └── `train.target.txt`     
          └── `val.source.txt` 
          └── `val.target.txt` 
          └── `test.source.txt` 
          └── `test.target.txt` 
```

## Quick Start

### 1. Install packages
```shell
pip install -r requirements.txt
```
And you have to install **Pytorch** from their [homepage](https://pytorch.org/get-started/locally/).

### 2. Collect Datasets and Resources

As mentioned above.

### 3. Run the code for training or testing

Please refer to the command examples listed in `python_commands.sh`:

For example, for training **Bart**:
```shell
python tasks/tongue_twister/train.py --data_dir=datasets/tongue_twister/tt-prompt-data \
 --learning_rate=8e-5 --train_batch_size=32 --eval_batch_size=2 --model_name_or_path=facebook/bart-base \
 --output_dir=output/tongue_twister --model_name my_bart --experiment_name=tongue_twister-my_bart\
 --max_source_length 100 --max_target_length 150 \
 --val_check_interval=0.5 --limit_val_batches=1 --max_epochs=5 --accum_batches_args=1 --num_sanity_val_steps=1
```

```shell
python tasks/tongue_twister/test.py\
  --eval_batch_size=64 --model_name_or_path=facebook/bart-base \
  --output_dir=output/tongue_twister/ --model_name my_bart --experiment_name=tongue_twister-my_bart --eval_beams 4 \
  --max_target_length=150
```

Revise the parameters according to your demand.

## Notation
Nothing is difficult. For technical details please refer to our paper.

## Citation
If you found this repository or paper is helpful to you, please cite our paper. 
Currently we only have arxiv citation listed as follows:

This is the arxiv citation:
```angular2
@article{loakman2023phonetically,
  title={Phonetically-Grounded Language Generation: The Case of Tongue Twisters},
  author={Loakman, Tyler and Tang, Chen and Lin, Chenghua},
  journal={arXiv preprint arXiv:2306.03457},
  year={2023}
}
```


