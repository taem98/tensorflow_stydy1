# chapter5 텐서보드와 모델 재사용

> 학습시킨 모델을 저장하고 재사용 하는 방법과 텐서보드를 이용해 손실값의 변화를 그래프로 추적해보자

## 5.1 학습 모델 저장하고 재사용하기

앞선 코드를 쓰지만 데이터를 scv파일로 분리

```python
#data.csv

# 털, 날개, 기타, 포유류, 조류

0, 0, 1, 0, 0
1, 0, 0, 1, 0
1, 1, 0, 0, 1
0, 0, 1, 0, 0
0, 0, 1, 0, 0
0, 1, 0, 0, 1
```

데이터를 읽어 들이고, 변환하는 코드

```python
import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',',
                    unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

```

- `unpack=True` : 열 --> 행
- `transpose` : 행으로 자른 데이터를 열 --> 행

<br/>
신경망 모델을 구성<br/>
먼저, 모델을 저장할 때 사용할 변수 만듦<br/>
이 변수는 학습에 직접 사용되지 않고, 학습 횟수를 카운트 하는 변수

```python
global_step = tf.Variable(0, trainable=False, name='global_step')
```

앞 장이 코드보다 계층은 하나 더 늘리고, 편향은 없고 가중치만 사용

```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,10], -1, 1))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10,20], -1, 1))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20,3], -1, 1))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model)
)

optimizer = tf.trainAdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)
```

- `global_step` 
    - 최적화 함수가 학습용 변수들을 최적화 할 때마다 global_step 변수의 값을 1씩 증가시킴

모델 구성 끝.<br/>
모델을 부르고 저장하는 코드

```python
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
```
- `global_variable` : 앞서 정의한 변수들을 가져오는 함수

<br/>
다음 코드는 ./model 디렉터리에 모델이 있는지 확인

- 모델이 있으면 
    - saver.restore 함수를 사용해서 학습된 값들을 불러오고
- 모델이 없다면
    - 변수를 새로 초기화 한다.

- 체크포인트 파일 
    - 학습된 모델을 저장한 파일

```python
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
```

- `global_step`
    - 텐서 타입의 변수
        - 값을 가져올 때 sess.run(global_step) 이용
    

```python
for step in range(2):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

    print('Step: %d,' % sess.run(global_step),
        'Cost: $.3f' % sess.run(cost, feed_dict={X:x_data, Y:y_data}))
```

최적화가 끝난 뒤 학습된 변수들을 지정한 체크포인트 파일에 저장

```python
saver.save(sess, './model/dnn.ckpt', global_step=global_step)
```

예측 결과와 정확도를 확인할 수 있는 다음 코드

```python
prediction = tf.argmax(model,1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:', sess.run(target, feed_dict={Y:x_data}))

is_correct = tf.equal(prediction, traget)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```

## 5.2 텐서보드 사용하기

> 텐서보드는 학습하는 중간중간 손실값이나 정확도 또는 결과물로 나온 이미지나 사운드 파일들을 다양한 방식으로 시각화해서 보여준다.

앞서 만든 코드에 텐서보드를 이용하기 위한 코드를 만들어보자!

```python

import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
```
그 다음으로 신경망의 각 계층에 다음 코드를 넣어준다.

```python
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2,10], -1, 1), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))
```

- `with tf.name_scope`
    - 묶은 블록은 텐서보드에서 한 계층 내부를 표현해줌

```python

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1, 1))
    L2 = tf.nn.relu(tf.matmul(L1, W2))
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20,3], -1, 1))
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
```

다음으로 손실값을 추적하기 위해 수집할 값을 지정하는 코드를 작성하자

```python
    tf.summary.scalar('cost', cost)
```

tf.summary 모듈의 scalar 함수는 값이 하나인 텐서를 수집할 때 사용 <br/>
물론 scalar 뿐만 아니라 histogram, image, audio 등을 다양한 값을 수집하는 함수를 기본으로 제공

<br/><br/>
이제 모델을 부르거나 초기화하는 코드를 넣고,

```python
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
```

 `tf.summary.merge_all` 함수로 지정한 텐서들을 수집한 다음 `tf.summary.FileWrite`함수를 이용해서 그래프와 텐서들의 값을 저장할 디렉토리 설정

```python
merged = tf.summary.merge_all()
writer = tf.summary.FileWirte('./logs', sess.graph)
```

그런 다음 최적화를 실행하는 코드를 앞서와 같이 작성하고,

```python
for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

    print('Step: %d,' % sess.run(global_step),
        'Cost: %.3f' % sess.run(cost, feed_dict={X:x_data, Y:y_data}))
```

sess.run을 이용해 앞서 merged로 모아둔 텐서의 값들을 계산하여 수집한 뒤, writer.add_summary함수를 이용해 해당 값들을 아서 지정한 디렉터리에 저장.
<br/>
적절한 시점에서 (여기서는 매 단계에서) 값들을 수집하고 저장하면 되며, 나중에 확인할 수 있도록 global_step값을 이용해 수집한 시점을 기록해둔다.

<br/><br/>마지막으로 예측부분

```python

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 :', sess.run(prediction, feed_dict={X:x_data}))
print('결과값 :', sess.run(target, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```
