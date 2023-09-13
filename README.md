# NMER

This repo implements the Noise-robust Multimodal Emotion Recognition(NMER)  model for the following paper:
"Learning Noise-Robust Joint Representation for Multimodal Emotion Recognition under Realistic Incomplete Data Scenarios".
You can obtain the NMER model in models/NMER_model.py, and the Noise scheduler in Domiss.py.
You can modify the Noise scheduler to fit your own data framework.
Hope you can enjoy the Neural Network!

# Reference

Thanks to the previous works, we used the code framework of  @JinmingZhao at https://github.com/JinmingZhao and @ZhuoYulang at https://github.com/ZhuoYulang to load and accelerate our model.

# Environment

``` 
python 3.9.0
pytorch >= 1.13.0
```

# Usage

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```.

You can download the preprocessed feature to run the code.

+ For Training NMER on IEMOCAP:

    First training the pretrained encoder with full modalities and complete data.

    ```bash
    bash scripts/CAP_utt_shared.sh AVL [num_of_expr] [GPU_index]
    ```

    Then train the NMER model.

    ```bash
    bash scripts/ours/CAP_NMER.sh [num_of_expr] [GPU_index]
    ```


Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/get_opt.py and the ```modify_commandline_options``` method of each model you choose.

# Download the features
Baidu Yun Link
IEMOCAP A V L modality Features
Link: https://pan.baidu.com/s/1WmuqNlvcs5XzLKfz5i4iqQ 
Extract code: gn6w 

