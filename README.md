<div align="center">
  <a href="https://github.com/evelyn0414/OPERA"> <img width="200px" height="200px" src="https://github.com/evelyn0414/OPERA/assets/61721952/6d17e3e7-5b3f-4e0b-991a-1cc02c5434dc"></a>
</div>


-----------------------------------------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)

This is the official code release for **OPERA**: **OPE**n **R**espiratory **A**coustic foundation models.

OPERA is an OPEn Respiratory Acoustic foundation model pretraining and benchmarking system. We curate large-scale respiratory audio datasets (136K samples, 440 hours), pretrain three pioneering foundation models, and build a benchmark consisting of 19 downstream respiratory health tasks for evaluation. Our pretrained models demonstrate superior performance (against existing acoustic models pretrained with general audio on 16 out of 19 tasks) and generalizability (to unseen datasets and new respiratory audio modalities). This highlights the great promise of respiratory acoustic foundation models and encourages more studies using OPERA as an open resource to accelerate research on respiratory audio for health.

![framework](https://github.com/evelyn0414/OPERA/assets/61721952/30c6ed72-1720-4c2e-9351-79d48f03d3a4)


To reproduce the results in our [paper](), develop your own foundation models, or deploy our pretrained models for downstream healthcare applications, please follow the guideline below.


## Installation

The environment with all the needed dependeciescan be easily created on a Linux machine by running:
```
git clone https://github.com/evelyn0414/OPERA.git
cd ./OPERA

conda env create --file environment.yml
sh ./prepare_env.sh
source ~/.bashrc

conda init
conda activate audio
sh ./prepare_code.sh
```

*After installation, next time to run the code, you only need to acivate the audio env by `conda activate audio`.



## Prepare data

| Dataset                                  | Source | Access                                                       | License        |
| ---------------------------------------- | ------ | ------------------------------------------------------------ | -------------- |
| UK COVID-19      | IC     | [https://zenodo.org/records/10043978](https://zenodo.org/records/10043978) | OGL 3.0        |
| COVID-19 Sounds      | UoC    | [https://covid-19-sounds.org/blog/neurips_dataset](https://covid-19-sounds.org/blog/neurips_dataset) | Custom license |
| CoughVID      | EPFL   | [https://zenodo.org/records/4048312](https://zenodo.org/records/4048312) | CC BY 4.0      |
| ICBHI                | *      | [https://bhichallenge.med.auth.gr](https://bhichallenge.med.auth.gr) | CC0            |
| HF Lung    | *      | [https://gitlab.com/techsupportHF/HF_Lung_V1](https://gitlab.com/techsupportHF/HF_Lung_V1) | CC BY-NC 4.0   |
|                                          |        | [https://gitlab.com/techsupportHF/HF_Lung_V1_IP](https://gitlab.com/techsupportHF/HF_Lung_V1_IP) |                |
| Coswara   | IISc   | [https://github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data) | CC BY 4.0      |
| KAUH           | KAUH   | [https://data.mendeley.com/datasets/jwyy9np4gv/3](https://data.mendeley.com/datasets/jwyy9np4gv/3) | CC BY 4.0      |
| Respiratory@TR | ITU    | [https://data.mendeley.com/datasets/p9z4h98s6j/1](https://data.mendeley.com/datasets/p9z4h98s6j/1) | CC BY 4.0      |
| SSBPR              | WHU    | [https://github.com/xiaoli1996/SSBPR](https://github.com/xiaoli1996/SSBPR) | CC BY 4.0      |
| MMlung               | UoS    | [https://github.com/MohammedMosuily/mmlung](https://github.com/MohammedMosuily/mmlung) | Custom license |
| NoseMic      | UoC    | [https://github.com/evelyn0414/OPERA/tree/main/datasets/nosemic](https://github.com/evelyn0414/OPERA/tree/main/datasets/nosemic)                                                           | Custom license |

*ICBHI and HF Lung datasets come from multiple sources. COVID-19 Sounds, SSBPR, MMLung and NoseMic are available upon request, while other data can be downloaded using the above url. Custom license is detailed in the DTA (data transfer agreement).

We provided some curated datasets which can be downloaded from the [Google drive]() (replace the `datasets` folder). 


## Pretraining foudation models using OPERA framework

Example training can be found in  `cola_pretraining.py` and `mae_pretraining.py`.

Start by running 

```
sh scripts/multiple_pretrain.sh
```

## Using OPERA models

The pretrained weights are available at:
__Zenodo__ or <a href="https://huggingface.co/evelyn0414/OPERA/tree/main" target="_blank"> HuggingFace </a>


our pretrained model checkpoints:
[OPERA-CT](https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaCT.ckpt?download=true), [OPERA-CE](https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaCE.ckpt?download=true), [OPERA-GT](https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaGT.ckpt?download=true).

They will be audomatically downloaded before feature extraction.


Run example of one task:
```
sh datasets/copd/download_data.sh
sh scripts/copd_eval.sh > cks/logs/Test_Task11_results.log
```
The log is included under 'cks/logs/' for reference. The results are listed below.  

| Task |  ID & Task Abbr | Opensmile     | VGGish       | AudioMAE      | CLAP  | **OPERA-CT** |     **OPERA-CE**  |**OPERA-GT**   |
| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------|
|T11   | COPD severity (Lung)   | 0.494 ± 0.054 | 0.590 ± 0.034 | 0.510 ± 0.021 | 0.636 ± 0.045   | 0.625 ± 0.038 | 0.683 ± 0.007 |0.606 ± 0.015   |       

## Run benchmark

 ```
sh scripts/benchmark.sh
```


## Citation

If you use OPERA, please consider citing:

```
```
