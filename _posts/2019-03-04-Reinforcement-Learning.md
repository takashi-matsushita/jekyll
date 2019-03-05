---
layout: post
categories: [AI,]
tags: [RL,]
title: Reinforcement Learning / 強化学習
author: Takashi MATSUSHITA
---

機械学習は、コンピュータ上での判断/予想を経験で学習/改善することだが、強化学習も機械学習の一方法.
ある状況下 (state[t]) で何らかの行動 (action[t]) を取った場合に報酬 (reward[t]) が得られる世界に、エージェントを送り込む. このエージェントは周囲の状況を観察 (observation[t]) し、様々な行動を取ることで報酬を最大化することを目的とする. 取るべき行動はポリシー (policy[t]) により決定するが、そのポリシーを経験により最適化 (update[t]) するのが強化学習.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="450" height="100" viewBox="0 0 450 100">
  {% include figures/reinforcement-learning.svg %}
</svg>
</div>

一連の学習過程はマルコフ決定過程 (Markov decision process) で以下の様に表される.
ここで、*R(S<sub>i-1</sub>,A<sub>i-1</sub>,S<sub>i-1</sub>)* は報酬 *R<sub>i-1</sub>* が、あるアクション *A<sub>i-1</sub>* で、状況が *S<sub>i-1</sub>* から *S<sub>i</sub>* に変化する際に得られることを表している. 取るべき行動は、ポリシー *&pi;(S<sub>i-1</sub>)* で決定される.

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="140" viewBox="0 0 800 140">
  {% include figures/MDP.svg %}
</svg>
</div>

強化学習用のツールである [Gym](https://gym.openai.com) を用いて強化学習の実際を見てみる.
経験によりポリシーを決定する仕組みを実装してみる. NChain-v0 というゲームは、あり得る状況が 5種類で、取り得る行動は 2種類. それぞれの状況下で取り得る行動を決定するポリシーを 5 x 2 の行列で表現する.
ポリシーが初期状態の場合には、取るべき行動を一様乱数を用いて決定する. ポリシーがアップデートされて居る場合には、報酬が多かった行動を取る.

```python
import gym
env = gym.make('NChain-v0')
nstate = env.observation_space.n
naction = env.action_space.n

def policy_table(env, nstate, naction, nplay=100):
  policy = np.zeros((nstate, naction))

  for _ in range(nplay):
    s0 = env.reset()
    done = False

    while not done:
      if np.sum(policy[s0, :]) == 0:
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(policy[s0, :])
      s1, reward, done, info = env.step(action)
      policy[s0, action] += reward
      s0 = s1

  return policy

policy = policy_table(env, nstate, naction)
```

これにより決定されたポリシーは以下の様になった.
```text
array([[15528.,     0.],
       [12118.,     0.],
       [    0., 39482.],
       [    0.,  7794.],
       [    0.,  4272.]])
```

次に将来予想される報酬を考慮して現在の行動を決定するために Q-value (state-action values) を最適化するアルゴリズムを使用する.

$$ Q_{k+1}(s,a) \leftarrow \Sigma_s T(s,a,s')[R(s,a,s')+\gamma\cdot\max_a Q_k(s',a')]$$

ここで、 $$T(s,a,s')$$ は 行動 $$a$$ を取った際の状態 $$s$$ から 状態 $$s'$$ への遷移確率.
$$R(s,a,s')$$ は、その際に得られる報酬. $$\gamma$$ は将来に期待される報酬の割引率.
ある状況下 *s* で取るべき行動は Q-value を最大化する行動、$$\pi^*(s) = \mathrm{argmax}_aQ^*(s,a)$$.
対応するコードは以下の様になる.

```python
def policy_table_qlearning(env, nstate, naction, nplay=100):
  ## initialise policy table
  policy = np.zeros((nstate, naction))

  y = 0.95  # discount factor
  lr = 0.8  # learning rate

  for _ in range(nplay):
    s0 = env.reset()
    done = False

    while not done:
      ## decide action to take
      if np.sum(policy[s0, :]) == 0:
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(policy[s0, :])

      ## update policy table, considering future reward
      s1, reward, done, info = env.step(action)
      policy[s0, action] += reward + lr*(y*np.max(policy[s1, :]) - policy[s0, action])
              
      s0 = s1

  return policy
```

これにより決定されたポリシーは以下の様になった.
```text
array([[21.05875701,  0.        ],
       [ 0.        , 22.22714818],
       [ 0.        , 22.796456  ],
       [ 0.        , 22.82361894],
       [ 0.        , 22.19951512]])
```

上記二つのアルゴリズム共に *S<sub>i</sub>* で最初に取る行動は一様乱数で決定される. 得られた報酬でポリシーを更新するが、一度報酬が得られた行動をその後ずっと踏襲する為に、*S<sub>i</sub>* で取る行動は一種類に限定される. 初めに乱数で決定された行動が、最善の行動かどうかは分からないので、このアルゴリズムによる最適化は不能.

次に、エージェントに好奇心を与えて、ある状況下でこれまでとは違う行動も取るようにしてみる.
*&epsilon;-greedy policy* と呼ばれる好奇心を与えたエージェントの学習アルゴリズムは以下の様になる.
ここで用いられている、decay はエージェントに飽きっぽさを与える. 何度も同じ事を繰り返していると面倒くさくなり此までの経験で学習したことを盲目的に適用するようになる.

```python
def policy_table_egreedy_qlearning(env, nstate, naction, nplay=100):
  ## initialise policy table
  policy = np.zeros((nstate, naction))

  y = 0.95      # discount factor
  lr = 0.8      # learning rate
  eps = 0.5     # epsilon
  decay = 0.999 # decay factor

  for _ in range(nplay):
    s0 = env.reset()
    done = False
    eps *= decay

    while not done:
      ## decide action to take
      if (np.random.random() < eps) or (np.sum(policy[s0, :]) == 0):
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(policy[s0, :])

      ## update policy table, considering future reward
      s1, reward, done, info = env.step(action)
      policy[s0, action] += reward + lr*(y*np.max(policy[s1, :]) - policy[s0, action])
              
      s0 = s1

  return policy
```

これにより決定されたポリシーは以下の様になった. *s<sub>i</sub>* で、取りうるべき行動全てが取られている.
この方法で学習回数を増やせば最適なポリシーが得られる.
```text
array([[79.04993073, 78.91555113],
       [82.93251651, 82.45432332],
       [86.74169249, 83.22698797],
       [82.09109268, 89.6518662 ],
       [97.24225729, 89.64617297]])
```

上記３つのアルゴリズムを対戦させてみる. 一度学習する毎に対戦し、計百回の学習と対戦を行う.
```python
def run_game(policy, env):
  s0 = env.reset()
  sum_reward = 0
  done = False
  while not done:
    action = np.argmax(policy[s0, :])
    s1, reward, done, info = env.step(action)
    sum_reward += reward
  return sum_reward


def compare_methods(env, nstate, naction, nplay=100):
  winner = np.zeros((3,))

  for _ in range(nplay):
    print('inf> loop = {}'.format(_))
    policy_rl = policy_table(env, nstate, naction)
    policy_qrl = policy_table_qlearning(env, nstate, naction)
    policy_egqrl = policy_table_egreedy_qlearning(env, nstate, naction)
    rl = run_game(policy_rl, env)
    qrl = run_game(policy_qrl, env)
    egqrl = run_game(policy_egqrl, env)
    w = np.argmax(np.array([rl, qrl, egqrl]))
    print('inf> winner = {}'.format(w))
    winner[w] += 1
  return winner

compare_algorithms(env, nstate, naction)
```

結果は、期待通り &epsilon;-greedy Q-Learning が 44% と最多勝率となった. (全てが同等の強さであれば勝率はそれぞれ 33% と予想される.)
```text
array([29., 27., 44.])
```


## Deep Q-Learning
Q-value を最適化する際に用いる多層ニューラルネットワークを Deep Q-Network (DQN) と呼ぶ.
また、DQN を用いた Q-Learning を Deep Q-Learning と呼ぶ.
*&epsilon;-greedy Q-Learning* を多層ニューラルネットワークを用いて実装してみる.

```python
def deep_q_learning(env, nstate, naction, nplay=100):
  ## build deep Q-network
  from keras.models import Sequential
  from keras.layers import InputLayer, Dense
  from keras.callbacks import CSVLogger

  path_log = 'rl_gym-log.csv'
  callbacks = [
    CSVLogger(filename=path_log, append=True),
    ]
  if os.path.isfile(path_log): os.remove(path_log)

  model = Sequential()
  model.add(InputLayer(batch_input_shape=(1, nstate)))
  model.add(Dense(10, activation='sigmoid'))
  model.add(Dense(naction, activation='linear'))
  model.compile(loss='mse', optimizer='adam', metrics=['mae'])


  y = 0.95      # discount factor
  eps = 0.5     # epsilon
  decay = 0.999 # decay factor

  sum_rewards = []
  for ii in range(nplay):
    s0 = env.reset()
    eps *= decay
    done = False
    if ii % 10 == 0:
      print("loop {} of {}".format(ii+1, nplay))

    sum_reward = 0.
    while not done:
      ## decide action to take
      if (np.random.random() < eps):
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(model.predict(np.identity(nstate)[s0:s0+1]))

      s1, reward, done, info = env.step(action)

      ## update deep Q-network
      target = reward + y * np.max(model.predict(np.identity(nstate)[s1:s1+1]))
      target_vec = model.predict(np.identity(nstate)[s0:s0 + 1])[0]
      target_vec[action] = target 
      model.fit(np.identity(nstate)[s0:s0+1], target_vec.reshape(-1, naction),
                callbacks=callbacks, epochs=1, verbose=0)
      s0 = s1
      sum_reward += reward
    sum_rewards.append(sum_reward)

  # construct policy table
  policy = np.zeros((nstate, naction))
  for ii in range(nstate):
    policy[ii] = model.predict(np.identity(nstate)[ii:ii+1])

  return policy, sum_rewards
```

学習回数毎の報酬の推移. 徐々に報酬が最大化されていくのが見て取れる.
<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="250" viewBox="0 0 600 500">
  {% include figures/rl_gym-DQL-hist.svg %}
</svg>
</div>

1000回の学習で得られたポリシーは以下の通り.
```text
array([[61.61214447, 60.6944046 ],
       [65.25592041, 61.80910492],
       [69.88792419, 63.04613495],
       [75.7440033 , 64.34770203],
       [84.44594574, 66.54193115]])
```
全ての状態に於いて行動 0 を取ることが報酬を最大化する事が分かる.
```python
def print_policy(policy):
  for ii in range(policy.shape[0]):
    print('state={} action={}'.format(ii, np.argmax(policy[ii, :])))

print_policy(policy_3)
state=0 action=0
state=1 action=0
state=2 action=0
state=3 action=0
state=4 action=0
```

今回使用したコードは[こちら](https://github.com/takashi-matsushita/lab/blob/master/dnn/rl_gym.py)
