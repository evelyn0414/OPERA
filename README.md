<div align="center">
  <a href="https://github.com/evelyn0414/OPERA"> <img width="200px" height="200px" src="https://github.com/evelyn0414/OPERA/assets/61721952/6d17e3e7-5b3f-4e0b-991a-1cc02c5434dc"></a>
</div>


-----------------------------------------
[![arXiv](https://img.shields.io/badge/arXiv-2406.16148-b31b1b.svg)](https://arxiv.org/abs/2406.16148)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/evelyn0414/OPERA)
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



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



## Preparing data

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

*ICBHI and HF Lung datasets come from multiple sources. COVID-19 Sounds, SSBPR and MMLung  are available upon request, while other data can be downloaded using the above url. Custom license is detailed in the DTA (data transfer agreement).

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

Run example of Task 10:
```
sh datasets/KAUH/download_data.sh
sh scripts/kauh_eval.sh > cks/logs/Test_Task10_results.log
```

Run example of Task 11:
```
sh datasets/copd/download_data.sh
sh scripts/copd_eval.sh > cks/logs/Test_Task11_results.log
```
The log is included under 'cks/logs/' for reference. The results for all tasks are summarised in Table 4 and 5.  
    

## Running the benchmark

 ```
sh scripts/benchmark.sh
```

## Understanding the model 

Run `res_analysis/saliency_map.py` for an analysis of the model using saliency maps.

|![Slide1](https://github.com/user-attachments/assets/ce83bdd8-943d-4dce-9bb1-1b2431cf9afd) | ![Slide2](https://github.com/user-attachments/assets/4689faad-d9ba-49ed-8248-561d64213362)|
|-------|-----------|
|![Slide3](https://github.com/user-attachments/assets/dbb9e910-05a8-4aaa-968e-8ffa79bf5869)|![Slide4](https://github.com/user-attachments/assets/19d31c45-27da-44f1-9974-06a213c03790)|
|![Slide5](https://github.com/user-attachments/assets/dcf72afd-07ed-4f80-b24a-96596a5f6136)|![Slide6](https://github.com/user-attachments/assets/89e8b4f4-68cb-4d37-8eed-0ca9843d17e7) |

## Citation

If you use OPERA, please consider citing:

```
@misc{zhang2024openrespiratoryacousticfoundation,
      title={Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking}, 
      author={Yuwei Zhang and Tong Xia and Jing Han and Yu Wu and Georgios Rizos and Yang Liu and Mohammed Mosuily and Jagmohan Chauhan and Cecilia Mascolo},
      year={2024},
      eprint={2406.16148},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2406.16148}, 
}
```
