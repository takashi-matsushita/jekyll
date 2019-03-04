---
layout: post
categories: [AI,]
title: Data Augmentation / データ拡張
author: Takashi MATSUSHITA
---

ひらがなのくずし字データを用いた画像認識では、適合率、再現率、F値の全てで 0.87 を得た. しかし、学習用データの統計が文字毎にかなり異なっていたので、今回はデータ拡張を用いて画像認識の向上を試みる. 目指すは 0.90.

とりあえず、全ての文字の学習用データ数が同じになるようにしてみる.
各文字毎に足りない分を補完するために元画像へのポインター (indices) を作る.

```python
### data augmentation, trying to have flat class distribution
classes, counts = np.unique(train_labels, return_counts=True)
idxs = np.where(counts < max(counts))[0]
idx_augment = []
np.random.seed(1013)
for idx in idxs:
  c_idxs = np.where(train_labels==idx)[0]
  nn = max(counts) - counts[idx]
  c_idxs = np.random.choice(c_idxs, nn)
  idx_augment.extend(c_idxs)
```

元画像からデータ拡張を行うために、ImageDataGenerator の設定を行う. 画像に対して回転と上下・左右の移動を行う.
```python
arguments = {
  'featurewise_center': False,
  'featurewise_std_normalization': False,
  'rotation_range': 20,
  'width_shift_range': 0.1,
  'height_shift_range': 0.1,
  'horizontal_flip': False,
  'vertical_flip': False,
  }
im_gen = image.ImageDataGenerator(**arguments)
```

ImageDataGenerator を用いてデータ拡張を実行し、文字毎に学習用データ数を揃えたデータセットを作成する.
```python
### prepare input set
X = np.expand_dims(train_images, axis=-1)
x_test = np.expand_dims(test_images, axis=-1)
y = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)

x_augmented = X[idx_augment].copy()
y_augmented = y[idx_augment].copy()
x_augmented = im_gen.flow(x_augmented,
                          batch_size=n_augment, shuffle=False).next()

# prepare full set
X = np.concatenate((X, x_augmented))
y = np.concatenate((y, y_augmented))
```

母集団の分布に従い訓練用データと確認用データに分割し、確認用データの文字毎の分布を見てみると、思惑通り一様分布になった.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 800 600">
  {% include figures/cnn-k49-augment-sample.svg %}
</svg>
</div>

前回学習済みのモデルを用いて継続して学習を実行してみる. 再学習にあたり、学習率を元に戻した.
```python
### load the pre-trained model
model = load_model('cnn-k49-last.h5')
# reset learning rate
K.set_value(model.optimizer.lr, 1.0)
```

因みに、以下のように畳み込み層を固定した転移学習を試してみたが、性能は改善しなかった. 畳み込み層を固定すると単位 epoch 当たりの実効時間はかなり短縮される.
```python
### transfer learning
for layer in model.layers:
  if layer.name.find('conv') != -1:
    layer.trainable = False
  if layer.name.find('pooling') != -1:
    layer.trainable = False

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

学習過程を見てみると、epoch=6 付近から顕著な学習の進展は見られない.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 600 450">
  {% include figures/cnn-k49-augment-hist.svg %}
</svg>
</div>

テストデータで学習成果を確認する.
```python
predict_classes = model.predict_classes(x_test)
print(classification_report(y_true, predict_classes)
```
適合率、再現率、F値は全て 0.93、0.93、0.93 と、各6％の改善が見られた.

混同行列を見てみると、対角成分の値が小さい場合にある特定の文字との間違いが多いように見受けられる箇所もある. この点に注目してデータ拡張を行えばもう少し性能が上がるかも知れないが、GPU が無いと色々と試すのに時間が掛かる.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 600 450">
  {% include figures/cnn-k49-augment-cm.svg %}
</svg>
</div>

F1値が 下位から５番目までの文字を拾ってみると、'え' (0.77)、'せ' (0.87)、'ほ' (0.87)、'ゐ' (0.83)、'ゑ' (0.82) となっている. これらの文字が誤認識された図を表示してみると、学習用データを様々な字体を含めてサンプル数を増やす事が良いように思われる. 図中、キャプション &lt;A: B&gt; の A は正解、B は予想.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 600 450">
  {% include figures/cnn-k49-augment-poor.svg %}
</svg>
</div>

使用したコードは[こちら](https://github.com/takashi-matsushita/lab/blob/master/dnn/cnn-k49-augment.py).

