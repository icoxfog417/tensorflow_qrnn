# Tensorflow QRNN

QRNN implementation for TensorFlow. Implementation refer to below blog.

[New neural network building block allows faster and more accurate text understanding](http://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/)

![qrnn.PNG](./qrnn.PNG)

## Dependencies

* TensorFlow: 0.12.0rc0
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


## Experiments

```
Baseline Working check
Iter 0: loss=2.4843482971191406, accuracy=0.1796875
Iter 100: loss=0.3832557201385498, accuracy=0.859375
Iter 200: loss=0.10648348927497864, accuracy=0.9765625
Iter 300: loss=0.053328096866607666, accuracy=0.9765625
Iter 400: loss=0.04552268981933594, accuracy=0.984375
Iter 500: loss=0.03381854295730591, accuracy=0.984375
Iter 600: loss=0.016251254826784134, accuracy=1.0
Iter 700: loss=0.007536895107477903, accuracy=1.0
Iter 800: loss=0.0045622605830430984, accuracy=1.0
Iter 900: loss=0.0045885248109698296, accuracy=1.0
Testset Accuracy=0.953125
```

```
QRNN Working check
Iter 0: loss=6.474578857421875, accuracy=0.078125
Iter 100: loss=1.707556128501892, accuracy=0.5234375
Iter 200: loss=0.7184199094772339, accuracy=0.78125
Iter 300: loss=0.6150789260864258, accuracy=0.8125
Iter 400: loss=0.312747061252594, accuracy=0.90625
Iter 500: loss=0.2874012589454651, accuracy=0.890625
Iter 600: loss=0.23133453726768494, accuracy=0.9375
Iter 700: loss=0.13557294011116028, accuracy=0.96875
Iter 800: loss=0.13632610440254211, accuracy=0.9609375
Iter 900: loss=0.07524901628494263, accuracy=0.9921875
Testset Accuracy=0.9296875
```

