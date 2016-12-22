# Tensorflow QRNN

QRNN implementation for TensorFlow. Implementation refer to below blog.

[New neural network building block allows faster and more accurate text understanding](http://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/)

![qrnn.PNG](./pictures/qrnn.PNG)

## Dependencies

* TensorFlow: 0.12.0
* scikit-learn: 0.18.1 (for working check)

## How to run

**Forward Test**

To confirm forward propagation, run below script.

```
python test_tf_qrnn_forward.py
```

**Working Check**

To confirm the performance of QRNN compare with baseline(LSTM), run below script.
Dataset is [scikit-learn's digit dataset](http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html).


```
python test_tf_qrnn_work.py
```

You can check the calculation result by [TensorBoard](https://www.tensorflow.org/versions/r0.12/how_tos/summaries_and_tensorboard/index.html).

![tensorboard.PNG](./pictures/tensorboard.PNG)

For example.

```
tensorboard --logdir=./summary/qrnn
```

## Experiments

```
Baseline(LSTM) Working check
Iter 0: loss=2.473149299621582, accuracy=0.1171875
Iter 100: loss=0.31235527992248535, accuracy=0.921875
Iter 200: loss=0.1704500913619995, accuracy=0.9453125
Iter 300: loss=0.0782063901424408, accuracy=0.9765625
Iter 400: loss=0.04097321629524231, accuracy=1.0
Iter 500: loss=0.023687714710831642, accuracy=0.9921875
Iter 600: loss=0.07718617469072342, accuracy=0.9765625
Iter 700: loss=0.02005828730762005, accuracy=0.9921875
Iter 800: loss=0.006271282210946083, accuracy=1.0
Iter 900: loss=0.007853344082832336, accuracy=1.0
Testset Accuracy=0.9375
takes 15.83749008178711 seconds.
```

```
QRNN Working check
Iter 0: loss=6.942812919616699, accuracy=0.0703125
Iter 100: loss=1.6366937160491943, accuracy=0.59375
Iter 200: loss=0.7058627605438232, accuracy=0.796875
Iter 300: loss=0.3940553069114685, accuracy=0.8984375
Iter 400: loss=0.2623080909252167, accuracy=0.9375
Iter 500: loss=0.3940059542655945, accuracy=0.921875
Iter 600: loss=0.1395827978849411, accuracy=0.96875
Iter 700: loss=0.11944477260112762, accuracy=0.984375
Iter 800: loss=0.1389300674200058, accuracy=0.9765625
Iter 900: loss=0.09582504630088806, accuracy=0.96875
Testset Accuracy=0.9140625
takes 13.540465116500854 seconds.
```
