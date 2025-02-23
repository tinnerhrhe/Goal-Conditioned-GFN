# Goal-Condtioned GFlowNets | ICLR 2025
This is the official code of the paper ["Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets"](https://openreview.net/forum?id=fNMKqyvuZT). we propose a novel method called **R**etrospective **B**ackward **S**ynthesis (**RBS**) to synthesize new backward trajectories in goal-conditioned GFlowNets to enrich training trajectories with enhanced *quality* and *diversity*, thereby introducing copious learnable signals for effectively tackling the sparse reward problem.
## Environment Configurations
```
conda env create -f environment.yml
conda activate gflownet
```
## Instructions
- Run RBS GFlowNets on the Grid tasks using
```
cd grid
python main.py --method db_gfn --batch_size 256 --backward 1 --horizon 32 --seed 1 --exp_name <name> --device cuda:0 --wdb
```
`horizon=32, 64, 128` denotes small, medium and large grid maps respectively.
- Run RBS GFlowNets on the bit sequence generation tasks using
```
cd gfn_bit
python main.py --seq_max_len 40 --num_bits 4 --fl 1 --device cuda:0 --wdb 1 --exp_name <name> --batch_size 256
```
- For small bit size: seq_max_len=40, num_bits=2
- For medium bit size: seq_max_len=60, num_bits=3
- For large bit size: seq_max_len=100, num_bits=5
## Citation
```bibtex
@inproceedings{
he2025looking,
title={Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned {GF}lowNets},
author={Haoran He and Can Chang and Huazhe Xu and Ling Pan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
}
```