# Automated Learning Rate Scheduling with In-Training Information

This repository contains the final report and presentation slides for the project **"Automated Learning Rate Scheduling with In-Training Information"**.

## Overview

Choosing an appropriate learning rate is one of the most critical and challenging tasks in training deep neural networks. Traditional schedulers like linear, cosine, or step decay are predetermined and do not flexibly adapt to model or dataset-specific dynamics.

This project proposes a **flexible, automatic learning rate scheduler** that adjusts the learning rate based on **in-training information**, specifically training loss.

## Key Idea

- Inspired by the Adaptive Learning Rate Tree (ALR) method, but improved for **lower computational cost**.
- Calculates training losses using multiple scale factors and uses a **weighted geometric mean** to determine the next learning rate.
- The scoring is based on normalized training losses using an exponential function.

## Method Highlights

- Requires only **N forward-backward passes** (instead of M×N in ALR).
- Can be applied across datasets and architectures.
- Uses ResNet18 on CIFAR-10 and CIFAR-100 for evaluation.

## Results

- Outperforms traditional schedulers (e.g., cosine, linear) on CIFAR-10.
- More stable generalization when using smaller batch sizes for factor scoring.
- Shows some limitations when starting from extreme learning rates.
