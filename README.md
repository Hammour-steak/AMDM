<h1 align="center"> Aggregation of Multi Diffusion Models for Enhancing Learned Representations </h1>

Aggregation of Multi Diffusion Models (AMDM) algorithm can aggregate features from different conditional control diffusion models with the same theoretical foundations into a specific model, enabling fine-grained conditional control.

In this GitHub repository, we provide three examples to verify the effectiveness of the algorithm.


# How to use
1. Installing conda environment.
    ```
    conda create --name AMDM python=3.8
    conda activate AMDM
    pip install -r requirements.txt
    ```
2. Prepare corresponding checkpoint from hugginface and setting path in configs.

3. Runing **xxxx_ui.py**. For example: `python interactmigc_ui.py`.


Thanks to [InteractDiffusion](https://github.com/jiuntian/interactdiffusion), [MIGC](https://github.com/limuloo/MIGC) and [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).