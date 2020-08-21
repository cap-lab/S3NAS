# S<sup>3</sup>NAS: Fast NPU-aware Neural Architecture Search
We conduct NAS Following three steps : **S**upernet design, **S**ingle-Path NAS with modification, **S**caling

## Requirements
* Access to Cloud TPUs ([Official Cloud TPU Tutorial](https://cloud.google.com/tpu/docs/tutorials/mnasnet))
* Tensorflow 1.13+
* Python 3.5+

### To run the experiments:

1. Set up ImageNet dataset

    To setup the ImageNet follow the instructions from [here](https://cloud.google.com/tpu/docs/imagenet-setup)  
    
    Or you can just copy from other bucket using `gsutil -m cp -r`, or transfer from other bucket.

2. Set up the profiled latency files
    ```
    latency_folder
    |-- Conv2D
    |-- Dense
    |-- GlobalAvgPool
    |-- MBConvBlock
    |-- MixConvBlock
        |-- r1_k3,5_s22_e2,4_i32_o32_c100_noskip_relu_imgsize112
        |-- ...
    ```
    each latency file contains a dictionary with latency value. For example, the content of
    `r1_k3,5_s22_e2,4_i32_o32_c100_noskip_relu_imgsize112` may be `{"latency": 364425}`
    
    to use our profiled latency files for MIDAP, please type
    ```
    git submodule --init --recursive
   ```
    
3. Set up flags and run
    Refer to the script files in base_experiment_scripts, or set up flags yourself.
    When you use scripts in base_experiment_scripts, please MODIFY
    * Google Cloud Storage Bucket
    * Model file name
    * Google Cloud TPU name
    * Target latency
    * Latency folder name
    
    We provide script templates for NAS / train / post_process

4. Run the script file.
