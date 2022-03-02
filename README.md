# BEVT: BERT Pretraining of Video Transformers

Rui Wang<sup>1</sup>, Dongdong Chen<sup>2</sup>, Zuxuan Wu<sup>1</sup>, Yinpeng Chen<sup>2</sup>, Xiyang Dai<sup>2</sup>, Mengchen Liu<sup>2</sup>, Yu-Gang Jiang<sup>1</sup>, Luowei Zhou<sup>2</sup>, Lu Yuan<sup>2</sup> <br>
<sup>1</sup>Shanghai Key Lab of Intelligent Information Processing, School of Computer Science, Fudan Univeristy, <sup>2</sup>Microsoft Cloud + AI

> This repository hosts the official PyTorch implementation of the paper: "[**BEVT: BERT Pretraining of Video Transformers**](https://arxiv.org/abs/2112.01529)".

## Abstract

This paper studies the BERT pretraining of video transformers. It is a straightforward but worth-studying extension given the recent success from BERT pretraining of image transformers. We introduce BEVT which decouples video representation learning into spatial representation learning and temporal dynamics learning. In particular, BEVT first performs masked image modeling on image data, and then conducts masked image modeling jointly with masked video modeling on video data. This design is motivated by two observations: 1) transformers learned on image datasets provide decent spatial priors that can ease the learning of video transformers, which are often times computationally-intensive if trained from scratch; 2) discriminative clues, i.e., spatial and temporal information, needed to make correct predictions vary among different videos  due to large intra-class and inter-class variations. We conduct extensive experiments on three challenging video benchmarks where BEVT achieves very promising results. On Kinetics 400, for which recognition mostly relies on discriminative spatial representations, BEVT achieves comparable results to strong supervised baselines. On Something-Something-V2 and Diving 48, which contain videos relying on temporal dynamics, BEVT outperforms by clear margins all alternative baselines and achieves state-of-the-art performance with a 71.4% and 87.2% Top-1 accuracy respectively.



<img src="assets/bevt_framework.png">



## Main Results on Downstream Tasks

**Something-Something V2**

| Backbone |  Pretrain   | Tokenizer | acc@1 | #params | FLOPs | Views | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | ImageNet-1K + K400 |  DALL-E   |  70.6  |   89M   |  321G  |  1x3  |  ToDo  | ToDo |
|  Swin-B  | ImageNet-1K + K400 |  PeCo     |  71.4  |   89M   |  321G  |  1x3  |  ToDo  | ToDo |


**Kinetics-400**

| Backbone |  Pretrain   | Tokenizer | acc@1 | #params | FLOPs | Views | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | ImageNet-1K + K400 |  DALL-E   |  80.6  |   88M   |  282G  |  4x3  |  ToDo  | ToDo |
|  Swin-B  | ImageNet-1K + K400 |  PeCo     |  81.5  |   88M   |  282G  |  4x3  |  ToDo  | ToDo |

**Note**:

- BEVT uses the visual tokenizer of pretrained VQ-VAE from [DALL-E](https://arxiv.org/abs/2102.12092) or [PeCo](https://arxiv.org/abs/2111.12710).
- PeCo is only pretrained on ImageNet1K and uses the same codebook size as in DALL-E.
- BEVT does not need labels during pretraining.


## To Do
- [ ] Release pretraining code
- [ ] Release fine-tuning code  
- [ ] Release pretrained model


## Citation

```
@article{wang2021bevt,
  title={BEVT: BERT Pretraining of Video Transformers},
  author={Wang, Rui and Chen, Dongdong and Wu, Zuxuan and Chen, Yinpeng and Dai, Xiyang and Liu, Mengchen and Jiang, Yu-Gang and Zhou, Luowei and Yuan, Lu},
  journal={arXiv preprint arXiv:2112.01529},
  year={2021}
}

@article{dong2021peco,
  title={PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers},
  author={Dong, Xiaoyi and Bao, Jianmin and Zhang, Ting and Chen, Dongdong and Zhang, Weiming and Yuan, Lu and Chen, Dong and Wen, Fang and Yu, Nenghai},
  journal={arXiv preprint arXiv:2111.12710},
  year={2021}
}
```
