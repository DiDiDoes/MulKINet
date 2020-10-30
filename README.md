# MulKINet
Multi-stage key-invariant CNN for accurate and fast cover song identification

Conference paper submitted to ISSPIT2020



## Introduction

Cover song identification (CSI) is a challenging task in the music information retrieval (MIR) community.The employment of convolutional neural networks (CNN) have significantly improved the performance of CSI systems, especially CNN designed to be invariant against key transpositions.

One important element for key-invariant CNN is frequential reception field 12. To achieve an equivalent frequential reception field of 12, we use stack of multiple stages, each with a smaller frequential reception field. We follow the principle that every stage expands the frequential reception field by a constant number. Thus, the number of stages in key-invariant CNN must be a factor of 12. Therefore, all possible choices are 1, 2, 3, 4, 6, and 12. Denote the number of stage(s) as S, we name the corresponding CNN architecture MulKINet-S.

An illustration of our network architecture can be seen in the following figure:

![architecture](https://github.com/DiDiDoes/MulKINet/blob/main/figures/architecture.JPG)



## Environment

Python==3.7.3

tensorflow-gpu==1.10.1

numpy==1.14.3



## Dataset

Second Hand Song 100K2, hpcp feature (npy files) and list available from [this repository](https://github.com/NovaFrost/SHS100K2)

Covers80, mp3 files available from [this website](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/). Scripts to generate hpcp feature is available in the SHS100K2 repository.

All npy files should be stored in a single folder and be names as `<song_id>_<version_id>.npy`. List files should be provided in the following format:

```
<song_id1>	<version_id1>
<song_id2>	<version_id2>
...
```

An example of data arrangement：

```
MulKINet
 |-- meta
 |   |-- Covers80
 |   |-- SHS100K-TRAIN
 |   |-- SHS100K-VAL
 |   `-- SHS100K-TEST
 `-- data
     |-- covers80_hpcp_npy
     `-- youtube_hpcp_npy
```



## Training

Before first run, please create two new directories `models` and `log` under the root directory.

Run  train.py to initialize training.

```
Important options:
	--tag			Tag for this experiment
	--data-dir		Directory of dataset
	--train-ls		List of training set
	--val-ls		List of validation set
	--block			Building block: simple / bottleneck / wider
	--ki-block-num	        Number of key-invariant blocks
	--no-chnlatt	        Disable channel attention
	--no-tempatt	        Disable temporal attention
	--batchsize		Batch size for training
	--max-epoch		Max number of training epochs
	--gpu			ID of GPU(s) to use
```

Checkpoints will be saved to `models/<tag>`/ and logs will be saved to `log/<tag>/`.



## Evaluation

Run evaluation.py for evaluation.

```
Important options:
	--data-dir		Directory of dataset
	--test-ls		List of testing set
	--model file	        Model file to evaluate
	--ki-block-num	        Number of key-invariant blocks
	--gpu			ID of GPU(s) to use
```

