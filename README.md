# Deep Learning Codes

This repository is used to implement some deep learning models I've read in the papers or books, or just come up in my mind.

The models are mainly implemented using tf.keras 

## Quick Start

Using RunTask.py (Modify it, change working directory and system path to your own)

example:`python RunTask.py gru4rec`

## Already implemented models(updating)

* [2019-9-16] *Transfer Learning* : [Siamese Neural Networks for One-shot Image Recognition[ICML2015]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), main code in Tasks/TransferLearning/OneShotLearning_Omniglot.py
* [2019-9-26] *Session Recommendation* : [A Simple Convolutional Generative Network for Next Item Recommendation[WSDM 2018]]()，main code in Tasks/SessionRecommendation/SimpleConvNetForNextItem.py
* [2019-10-11] *Session Recommendation*：[Recurrent Neural Networks with Top-k Gains for Session-based Recommendations [ACM CIKM 2018]](https://arxiv.org/abs/1706.03847)
* [2019-10-14] *Image Generation*：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[ICLR 2016]](https://arxiv.org/abs/1511.06434)

# Datasets Added

* Omniglot Dataset(Handwriting symbols from 26 languages)
* Labeled Faces in the Wild(Deepfunneled, deleted folders with only one picture)
* Movielens(Latest-100k and 1M from official website)
* Anime Faces (6000+ anime faces from internet, width and height same and larger than 10KB)
* YooChoose-Click(YouChoose click data in sequence representation)