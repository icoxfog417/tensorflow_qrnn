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

Now, baseline is stronger... I'm very welcome to your advice!

```
Baseline Working check
Iter 0: loss=2.8711295127868652, accuracy=0.0859375
Iter 100: loss=0.313383013010025, accuracy=0.8828125
Iter 200: loss=0.2098013460636139, accuracy=0.9375
Iter 300: loss=0.048367783427238464, accuracy=1.0
Iter 400: loss=0.026696903631091118, accuracy=1.0
Iter 500: loss=0.006171069107949734, accuracy=1.0
Iter 600: loss=0.013685302808880806, accuracy=1.0
Iter 700: loss=0.006732741370797157, accuracy=1.0
Iter 800: loss=0.008468863554298878, accuracy=1.0
Iter 900: loss=0.001918797381222248, accuracy=1.0
Testset Accuracy=0.9375
```

```
QRNN Working check
Iter 0: loss=5.390337944030762, accuracy=0.109375
Iter 100: loss=1.474597454071045, accuracy=0.53125
Iter 200: loss=0.8689780235290527, accuracy=0.7109375
Iter 300: loss=0.7574909925460815, accuracy=0.703125
Iter 400: loss=0.5692278146743774, accuracy=0.796875
Iter 500: loss=0.6652932167053223, accuracy=0.7578125
Iter 600: loss=0.5059253573417664, accuracy=0.8203125
Iter 700: loss=0.575552225112915, accuracy=0.8125
Iter 800: loss=0.4823388159275055, accuracy=0.828125
Iter 900: loss=0.5090895295143127, accuracy=0.84375
Testset Accuracy=0.7421875
```

