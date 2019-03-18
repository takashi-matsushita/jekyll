---
layout: post
categories: [AI,]
tags: [RL,Gym,Keras]
title: RL in discrete action space / 強化学習 - 離散化された行動空間 -
author: Takashi MATSUSHITA
---

行動空間が離散化された環境下での強化学習の再訪.
前回実装した DDPG の要素技術である、ポリシー最適化と Q-value の最適化を別々に見てみる. Gym の環境には CartPole-v0 を用いる.

まず、勾配上昇を用いたポリシーの最適化を行う.
ポリシーの勾配は、以下のように、ある行動 (a<sub>t</sub>) と 状態 (s<sub>t</sub>) から得られるポリシーの対数に報酬を掛けた物を微分する事で得られる.

$$\nabla_\theta J(\pi_\theta) = \mathrm{E}_{\tau\sim\pi_\theta}[\Sigma_{t=0}^{T}\nabla_\theta \log\pi_\theta(\alpha_t|s_t) \Sigma_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})] $$

ここで、

$$ \hat{R_{t}} = \Sigma_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) $$

は、reward-to-go (ある時点 (t') から先に得られる報酬). 損失関数は以下の Python コードで表される.

```python
def loss(y_true, y_pred):
  ## loss function
  #  Sum[ log(pi[a|s]) ] * R[tau]
  log_action_prob = K.sum(K.log(y_pred) * action_onehot, axis=1)
  return -K.mean(log_action_prob*rtg)
```

Q-value の最適化は既に実装したが、今回は double Deep Q-Network を使用してより安定した学習が出来るようにする. target の更新には、一定の頻度による更新と、soft update を用いた更新を試してみた. 更には、soft update の double Deep Q-Network に prioritized experience replay (PER) を適用してみた.

それぞれのアルゴリズムを比較するために、過去 100 epoch の平均報酬の推移を図にしてみる. 過去 100 epoch の平均報酬が 195 を超えた時点で試行を打ち切っている. ポリシー最適化、double DQN、soft update を用いた double DQN、PER を用いた double DQN の順に、より短期間で学習が実行できている.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="250" viewBox="0 0 600 500">
  {% include figures/CartPole.svg %}
</svg>
</div>

[CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0) に対して double DQN/PER を試した学習結果は以下のようになった.

<div align="center">
![CartPole-v0]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/CartPole.gif){: style="max-width: 300px; height: auto;"}
</div>


今回用いたコード
  * [policy gradient](https://github.com/takashi-matsushita/lab/blob/master/dnn/policy_gradient.py)
  * [double DQN](https://github.com/takashi-matsushita/lab/blob/master/dnn/ddqn.py)
  * [double DQN/PER](https://github.com/takashi-matsushita/lab/blob/master/dnn/ddqn-per.py)
