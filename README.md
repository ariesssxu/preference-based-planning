# Learning to Plan with Personalized Preferences

Manjie Xu*, Xinyi Yang*, Wei Liang✉, Chi Zhang✉ , Yixin Zhu✉  
(* equal contribution, ✉ corresponding authors)  

<a href='http://arxiv.org/abs/2502.00858'>
  <img src='https://img.shields.io/badge/Paper-Arxiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper Arxiv'>
</a>
<a href='https://sites.google.com/view/personalized-planning/home'>
  <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
</a> 
<a href='https://player.vimeo.com/video/1052700883?h=dfd0c7152d'>
  <img src='https://img.shields.io/badge/Project-Demo-red?style=plastic&logo=Youtube&logoColor=red' alt='Project Page'>
</a>

<div align=center>
  <img src=./assets/PbP-intro.png />
</div>

Effective integration of AI agents into daily life requires them to understand and adapt to individual human preferences, particularly in collaborative roles. We develop agents that not only learn preferences from few demonstrations but also learn to adapt their planning strategies based on these preferences.

## Checklist
- ✅ Environment Setup
- ✅ Benchmark Generation
- ✅ Baseline Implementation
- 🕳️ Plannning & Demo Generation
- 🕳️ Robot Interface

## Code Structure

```
PbP
├── baselines          # Code for replicating baselines
│     ├── EILEV     
│     ├── GPT      
│     ├── Llama3        
│     ├── LLaVA-NeXT  
│     ├── opt
│     └── run_gui.py
├── OmniGibson         # The simulating platform built upon NVIDIA's Omniverse
├── benckmark
│     ├── draw  
│     ├── examples      
│     ├── level_0      # examples of the pre-defined level_x preferences
│     ├── level_1  
│     ├── level_2
│     ├── ...  
│     ├── action.py    # some definition of primitive actions
│     ├── scene_config.yaml    # config of the scene in Omniverse
│     └── video_recoder.py     # the recorder of the demo samples
├── LICENCE
├── README.md
└── setup.py
```
## Install

You should firstly install [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/) and [Omnigibson](https://github.com/StanfordVL/OmniGibson) based on the following tutorials:
```
NVIDIA Omniverse: https://docs.omniverse.nvidia.com/install-guide/latest/index.html  
Omnigibson: https://behavior.stanford.edu/omnigibson/getting_started/installation.html
```
Our code is developed on the version 2023.1.1 of Isaac Sim. We recommend using a machine with RAM 32GB+ and NVIDIA RTX 2070+ following OmniGibson's requirements. The code can be run on a headless machine via docker, but we recommend using a machine with a display.

After successfully install Omnigibson, test on some demos:
```
# simple navigation
python benchmark/examples/nav_demo.py
# pick and place
python benchmark/examples/pick_and_place.py
```

## Benchmark
We define human preference in three different levels. We provide code to help sample demos of these preference in different scenes. You can also modify the sampling code to generate more customed demos. 
<div align=center>
  <img src=./assets/preferences.png/>
</div>

```
cd level_x && python sample.py
```

where `sample.py` mainly consists the sampling loop, `preference.py` mainly consists the main preference logic, and `task.py` hosts the task environment.

We also provide a subset of our sampled dataset at [huggingface](https://huggingface.co/datasets/ariesssxu/pbp). (Note: This dataset is unfiltered and may include some crash demos caused by instability during rendering in Omnigibson. We recommend performing the rendering process independently for optimal results.) 

<div align=center>
  <img src=./assets/datapoint.png />
</div>
A demonstration includes an egocentric video of agent activity, a bird's-eye-view map tracking agent position, and frame-level action annotations. We also provide third-person view recordings for enhanced visualization. 

We prioritize the egocentric perspective for two reasons: 1) it offers a clear view with minimal occlusions, and 2) it aligns with human perception, facilitating transfer to real-world data from head-mounted devices.

## Baselines
We provide all our implementations of the baselines in the `baseline` folder. For each baseline, we recommend creating an independent conda environment to avoid conflict. Navigate to each folder to see the install steps.

## Ack
[NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)  
[Omnigibson](https://github.com/StanfordVL/OmniGibson)

## Cite us
```
@article{xu2024preference,
  title={Learning to Plan with Personalized Preferences},
  author={Xu, Manjie and Xinyi Yang and Wei Liang and Chi Zhang and Yixin Zhu},
  year={2024}
}
```
