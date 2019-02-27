---
layout: post
categories: [AI,]
title: XOR を NN を用いて実装する
author: Takashi MATSUSHITA
---

排他的論理和 (exclusive or / XOR) は、下表のように振る舞う論理演算.

| $$x_1$$ | $$x_2$$ | $y$ |
|:-------:|:-------:|:---:|
| 0 | 0 | 0   
| 0 | 1 | 1   
| 1 | 0 | 1   
| 1 | 1 | 0   

XOR を 多層パーセプトロンを用いて実装する.
実装には、Keras/TensorFlow を用いる [^1].

まずは、学習用のデータを準備する. 上記の表を NumPy の配列で表現.
```python
### XOR training data
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # input
y = np.array([[0],[1],[1],[0]])          # output
```

次にモデルを構築する. 隠れ層のパーセプトロンは二個、出力層のパーセプトロンは一個. それぞれ、活性化関数に tanh (双曲線正接関数)、シグモイド関数 [^2]を用いる.
入力値はバイアスを含めて３つ. 出力層への入力もバイアスを含めて３つとなる.

<div align="center">
{% include figures/MLP_XOR.svg %}
</div>

* 入力層: $$\mathbf{X} = [1, x1, x2]$$
* 隠れ層: $$\mathbf{Y}$$ はバイアス項と $$\tanh(\mathbf{W^1\cdot X^T})$$
* 出力層: $$h = \mathbf{W}^2\cdot\mathbf{Y}$$

モデルは以下の python コードで記述される.


```python
### MLP with two input variables (except for bias),
### a hidden layer with 2 perceptrons and output layer with 1 perceptron
model = Sequential()
model.add(Dense(2, input_dim=2, use_bias=True))  # hidden layer
model.add(Activation('tanh'))                    # hyperbolic tangent activation function
model.add(Dense(1, use_bias=True))               # output layer
model.add(Activation('sigmoid'))                 # sigmoid activation function
```

このモデルにある重み $$\mathbf{W^1}$$ と $$\mathbf{W^2}$$、都合９個のパラメターを学習させる. 学習とはとどのつまり、決められた入力に対する出力が予想される値に近づくようにすること. 予想される出力とモデルの出力の差を計算し、その差を最小化する. この際、確率的勾配降下法を用いてパラメターを最適化する. 差を計算する際には、交差エントロピーを使用.

```python
### training
sgd = SGD(lr=0.1)   # stochastic gradient descent optimiser with learning rate = 0.1
model.compile(loss='binary_crossentropy', optimizer=sgd)
hist = model.fit(X, y, batch_size=1, epochs=2000, verbose=1)
```

下図は学習につれて、差が小さくなる様子を示しており、学習の効果が上がっていることが分かる.

<div align="center">
{% include figures/xor_loss.svg %}
</div>

学習済みモデルを用いて、入力に対する応答を見てみる.
```python
print(model.predict(X))
```
```
[[0.00591211]
 [0.9967507 ]
 [0.9967655 ]
 [0.0062728 ]]
```
予想される出力は、[0, 1, 1, 0]、なので上手く学習出来ていることが分かる.
以下の様に適当な閾値を使用すれば、このモデルは排他的論理和として使用可能.
```python
threshold = 0.5
print((model.predict(X) > threshold).astype(np.int))
```
```
[[0]
 [1]
 [1]
 [0]]
```


[^1]: [python notebook](https://github.com/takashi-matsushita/lab/blob/master/dnn/xor.ipynb)  [Python 3.7.2 / Keras 2.2.4 / TensorFlow 1.12.0 を使用]
[^2]: $$ S(x) = \frac{1}{1 + e^{-x}}$$
