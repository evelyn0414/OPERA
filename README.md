<div align="center">
  <a href="https://github.com/evelyn0414/OPERA"> <img width="300px" height="300px" src="https://github.com/evelyn0414/OPERA/assets/61721952/6d17e3e7-5b3f-4e0b-991a-1cc02c5434dc"></a>
</div>


-----------------------------------------

OPERA is an OPEn Respiratory Acoustic foundation model pretraining and benchmarking system. We curate large-scale respiratory audio datasets (136K samples, 440 hours), pretrain three pioneering foundation models, and build a benchmark consisting of 19 downstream respiratory health tasks for evaluation. Our pretrained models demonstrate superior performance (against existing acoustic models pretrained with general audio on 16 out of 19 tasks) and generalizability (to unseen datasets and new respiratory audio modalities). This highlights the great promise of respiratory acoustic foundation models and encourages more studies using OPERA as an open resource to accelerate research on respiratory audio for health.

![framework](https://github.com/evelyn0414/OPERA/assets/61721952/30c6ed72-1720-4c2e-9351-79d48f03d3a4)


This is the official code release for **OPERA**: **OPE**n **R**espiratory **A**coustic foundation models. To reproduce the results in our [paper](), develop your own foundation models, or deploy our pretrained models for downstream healthcare applications, please follow the guideline below.


## Installing

The environment can be easily created by running:
```
conda env create --file environment.yml
```

Then you can activate the environment and prepare the other dependecies:

```
conda activate audio
git clone https://github.com/username/repository.git
sh ./prepare_env.sh
```






## Installing
The curated datasets can be accessed from the Google drive (replace the `datasets` folder). 


## Using OPERA models

The pretrained weights are available at:
__Zenodo__ or __HuggingFace__

## Pretraining using OPERA framework

Example training can be found in  `cola_pretraining.py` and `mae_pretraining.py`.

Start by running 

```
sh scripts/multiple_pretrain.sh
```

## Citation

If you use OPERA, please consider citing:

```
```
