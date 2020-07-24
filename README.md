# multiple model stacking

## Introduction

In some case, there are several models for certain task, which products probability logits. So we need to fuse these logits from different models to make final decisions.

In this repo, we provide some methods of fusion.

## Major features

1. fusion of CNN features

2. fusion of probability logits

3. fusion of CNN features and probability logits

## Benchmark

Supported backbones:
- [x] ./fusion_features_logits_with_rnn.py
- [x] ./fusion_logits.py
- [x] ./fusion_logits_with_rnn.py

## Todo

1. early fusion

2. ...

## Changelog


## Contact

This repo is currently maintained by Zhenyu Lin ([@linzhenyu](https://github.com/linzhenyuyuchen)).






