# TWMA

This repository contains the pre-release code for the TWMA method as presented in our paper, "[Enhancement of price trend trading strategies via image-induced importance weights](https://www.arxiv.org/abs/2408.08483)." 

## Environment

- **Main Settings:** Python 3.9 & Pytorch 1.11.0 & CUDA 10.2 & [torchcam 0.3.2](https://github.com/frgfm/torch-cam)
- **Minor Settings:** To be completed.

## Data Pipeline

| Script                     | Description                                                                 |
| :------------------------: | :--------------------------------------------------------------------------: |
| `build_image_dataset.py` | Plots stock price images and calculates labels.         |
| `split_dataset.py`      | Splits the built image dataset into training, validation and testing.  |

## Network

| Script                     | Description                                                                 |
| :------------------------: | :--------------------------------------------------------------------------: |
| `distributed_random_train.py`                 | Trains the ResNet "trader".                               |
| `dataset.py`            | Defines the dataset structure based on PyTorch.                              |
| `distributed_utils.py`             | Some useful functions for distributed learning.                                                 |
| `inference.py`     | Obtains triple-I weights from the trained models.  

## Reproduce Part of Empirical Results

```bash
# Ensure you have updated the data path and log directory in each file.

# Step 1: Construct features and labels
python data_pipe/build_image_dataset.py
python data_pipe/split_dataset.py

# Step 2: Train trader.
CUDA_VISIBLE_DEVICES=0,1,2,3 python distributed_random_train.py

# Step 3: Inference and QCM learning
python network/inference.py
```

## Cite
If you find this code helpful, please consider citing our paper:
```
@misc{zhu2024enhancementpricetrendtrading,
      title={Enhancement of price trend trading strategies via image-induced importance weights}, 
      author={Zhoufan Zhu and Ke Zhu},
      year={2024},
      eprint={2408.08483},
      archivePrefix={arXiv},
      primaryClass={q-fin.PM},
      url={https://arxiv.org/abs/2408.08483}, 
}
```
## Contact
Please feel free to raise an issue in this GitHub repository or email me if you have any questions or encounter any issues.