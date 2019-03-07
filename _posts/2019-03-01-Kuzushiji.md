---
layout: post
categories: [AI,]
tags: [CNN,Keras]
title: KMNIST / くずし字認識
author: Takashi MATSUSHITA
---

ひらがなの[くずし字データ](https://www.kaggle.com/anokas/kuzushiji)[^1]を用いた画像認識を実装してみる. 
使用したコードは[こちら](https://github.com/takashi-matsushita/lab/blob/master/dnn/cnn-k49.py).
今回用いるのは 49文字のデータセット. 
文字毎の訓練用画像数を確認すると最大６千、最小千以下とかなりのばらつきが見られる.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 800 600">
  {% include figures/cnn-k49-training-sample.svg %}
</svg>
</div>

Uniform Manifold Approximation and Projection [UMAP](https://github.com/lmcinnes/umap) を用いて学習用データの２次元分布を見ると、MNIST と比較して ごちゃごちゃしており分類は難しそう.

MNIST   |  Kuzushiji
:-:|:-:
![MNIST-UMAP]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/mnist-umap.png){: style="max-width: 400px; height: auto;"} | ![K49-UMAP]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/cnn-k49-umap.png){: style="max-width: 400px; height: auto;"}

ConvNet には MNIST 画像データで 99.25% の精度を達成した[モデル](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py) を流用する. 

* 32個の3x3の畳み込みフィルタ層
* 64個の3x3の畳み込みフィルタ層
* 2x2の最大プーリング層
* 256個と49個のパーセプトロンからなる全結合層

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="600" height="200" viewBox="0 0 1500 500">
  {% include figures/kmnist-cnn.svg %}
</svg>
</div>


Keras を用いたモデルの実装は以下の様になる. モデルの要約を確認すると学習すべき変数の数は約240万個.

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

学習過程を見てみると、epoch=12 付近から顕著な学習の進展は見られない. 過学習は無い模様. GPU の無い MacBook Air で 1 epoch 当たり 12分程かかる.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 600 450">
  {% include figures/cnn-k49-hist.svg %}
</svg>
</div>

テストデータで学習成果を確認する.
```python
predict_classes = model.predict_classes(x_test)
print(classification_report(y_true, predict_classes)
```
適合率、再現率、F値は全て 0.87、0.87、0.87 であった.

混同行列を見てみると、対角成分に色むらが見られる. 上手く訓練できた文字とそうでない文字があるということ.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 600 450">
  {% include figures/cnn-k49-cm.svg %}
</svg>
</div>

F1値が 0.8以下の文字を拾ってみると、'な' (0.80)、'ま' (0.75)、'る' (0.78)、'ゐ' (0.75)、'ゝ' (0.57) となっている. これらの文字を表示してみると判別は文脈が無ければ難しそう.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 600 450">
  {% include figures/cnn-k49-poor.svg %}
</svg>
</div>


[^1]: "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
