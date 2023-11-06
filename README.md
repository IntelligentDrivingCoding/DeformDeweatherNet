# Residual Deformable Convolution for Better Image De-weathering

Source code for paper "Residual Deformable Convolution for Better Image De-weathering".

## Abstract

Adverse weather conditions pose great challenges to computer vision tasks in the wild. Image de-weathering, which aims at removing weather degradations from videos and images, has hence accumulated huge popularity as a significant component of image restoration. Considering the computational efficiency for on-device applications, Autoencoder-based deep models are widely adopted for image degradation removal due to its excellent generalization and high compu- tational efficiency. However, for most of these models, parts of high-frequency information are inevitably lost in the hierarchical feature embedding modules in the encoder, while degraded features are unable to be effectively inhibited in the upsampling modules in the decoder, limiting the restoration performance. In this paper, we propose a multi-patch skip-forward structure for the encoder to deliver fine-grain features from shallow layers to deep layers, embedding more detailed semantics for feature learning. In the decoding process, the Resid- ual Deformable Convolutional module is developed to dynamically recover the degradation with spatial attention, achieving high-quality pixel-wise reconstruc- tion. Extensive experiments show that our model outperforms many recently proposed state-of-the-art works on both specific-task de-weathering, such as de-raining, de-snowing, and all-task de-weathering.

## Requirements

```bash
torch == 1.10.0
torchvision == 0.11.0
scikit-image == 0.16.2
timm
tensorboard
einops
```

## Usage

The following commands are examples of training the model
```bash
CUDA_VISIBLE_DEVICES=2 python train.py -train_dir "/dataset/public/raindrop/test" -test_dir "/dataset/public/raindrop/train" -rain_subdir "data" -gt_subdir "gt"
```

**Note**: The dataset is organized as follows:
```bash
train_images: ${train_dir}/${rain_subdir}
train_labels: ${train_dir}/${gt_subdir}
test_images: ${test_dir}/${rain_subdir}
test_labels: ${test_dir}/${gt_subdir}
```
