# chapter4 기본 신경망 구현

> 신경망 모델, 딥러닝인 심층 신경망, 즉 다층 신경망 구현

## 4.1 인공신경망의 작동원리

### 인공신경망?
> 뇌를 구성하는 신경세포, 즉 **뉴런**의 동작 원리에 기초한다.

#### 뉴런의 동작원리

- 가지돌기 --> 축삭돌기 --> 축삭말단
    - 축삭돌기를 지나는 동안 신호가 약해지거나, 약해서 축삭말단까지 전달되지 않거나, 또는 강하게 전달됨.

#### 인공지능의 동작원리

- X(입력) --> x W (가중치 곱) --> + b (편향 합) --> Sigmoid, ReLU(활성화 함수) --> y (출력)
    1. 입력값 X에 가중치(W)를 곱한다.
    2. 편향(b)를 더한다.
    3. 활성화 함수 (Sigmoid, ReLU)
    4. y(결과값) 만들어내기

여기서 적절한 W와 b의 값을 찾아내는 최적화 과정을 **학습** 또는 **훈련**이라고 한다.

```python
y = Sigmoid(X x W + b)
```

#### 활성화 함수란?
> 인공신경망을 통과해온 값을 최종적으로 어떤 값으로 만들지를 결정한다.

인공뉴런의 핵심 중에서도 가장 중요한 요소<br/>

- Sigmoid
- ReLU
    - 최근 많이 사용
    - 0보다 작으면 0, 0보다 크면 입력값을 그대로 출력
- tanh

#### 발전해온 알고리즘의 중심에는 역전파가 있었다.

> 출력층이 내놓은 결과의 오차를 신경망을 따라 입력층까지 역으로 전파하여 계산해나가는 방식

- 최적화가 빠르고 정확해졌다.
- 역전파 구현은 어렵지만 텐서플로가 제공해준다.

### 4.2 간단한 분류 모델 구현하기

#### 분류?
> 패턴을 파악해 여러종류로 구분하는 직업
- 사진이 고양이인지, 강아지인지, 자동차인지, 비행기인지 판단

#### 예제
> 털과 날개의 유무 기준으로 포유류과 조류를 구분하는 신경망 모델 구현, 이미지 대신 간단한 이진데이터

```python
import tensorflow as tf
import numpy as np
```
- numpy : 수치 해석용 python 라이브러리

```python
# 학습에 사용할 데이터 정의
# 털과 날개가 있으면 1, 없으면 0
# [털, 날개]

x_data = np.array(
    [[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]]
)
```
그 다음으로 각 개체가 실제로 어떤 종류인지를 나타내는 레이블(분류값) 데이터를 구성한다. <br/>
즉, 앞의 특징데이터의 각 개체가 포유류인지 조류인지, 아니면 제 3의 종류인지를 기록한 실제 결과값이다.
<br/><br/>
##### 레이블데이터?

> 원-핫 인코딩 이라는 특수한 형태로 구성

- 원-핫 인코딩
    - 데이터가 가질 수 있는 값들을 일렬로 나열한 배열을 만듦
    - 그 중 표현하려는 값을 뜻하는 인덱스의 원소만 1로 표기, 나머지는 0

```python
# 원-핫 인코딩 예시
기타 = [1,0,0]
포유류 = [0,1,0]
조류 = [0,0,1]
```
```python
# 특징 데이터와 연관지어 레이블을 구성해보자

y_data = np.array(  # [털, 날개]
    [1,0,0],        # [0,0]
    [0,1,0],        # [1,0]
    [0,0,1],        # [1,1]
    [1,0,0],        # [0,0]
    [1,0,0],        # [0,0]
    [0,0,1],        # [0,1]
)
```
이제 신경망 모델을 구성<br/>
X와 Y에 실측값을 넣어서 학습시킬 것이니, X와 Y는 플레이스홀더로 설정

```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
```
그 다음으로 가중치와 편향 변수를 설정<br/>
신경망은 이 변수들의 값을 여러가지로 바꿔가면서 X와 Y의 연관 관계를 학습하게 된다.

```python
W = tf.Variable(tf.random_uniform([2,3], -1., 1.))
b = tf.Variable(tf.zeros([3]))
```
- 가중치 변수 W : [입력층(특징수), 출력층(레이블 수)]의 구성인 [2,3]으로 설정
- 편향 변수 b : 레이블 수인 3개의 요소를 가진 변수
> 이 가중치를 곱하고 편향을 더한 결과를 활성화 함수인 ReLU에 적용하면 신경망 구성은 끝

```python
L = tf.add(tf.matmul(x, W), b)
L - tf.nn.relu(L)
```
신경망 구현은 이걸로 끝<br/><br/>

추가로, 신경망을 통해 나온 출력값을 softmax함수를 이용해서 사용하기 쉽게 다듬어줌

```python
model = tf.nn.softmax(L)
```
- softmax함수
    - 배열 내의 결과값들을 전체 합이 1이 되도록 만들어준다.
    - 전체가 1이니, 각각은 해당 결과의 활률로 해석 가능

```python
[8.04, 2.76, -6.52] --> [0.53, 0.24, 0.23]
```
이제 손실함수(비용함수) 작성 <br/>
- **교차 엔트로피** 함수를 사용
    - 교차 엔트로피 값은 예측값과 실제값 사이의 확률 분포 차이를 계산한 값

```python
# 기본 코드
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
```
이제 학습

```python
# 기본적인 경사하강법으로 최적화한다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로의 세션을 초기화한다.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행합니다.
for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

    # 학습 도중 10번에 한 번씩 손실값을 출력해봅니다.
    if (step+1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

```
신경망을 구성하고 학습을 진행하는 전체 코드 끝<br/><br/>

마지막으로, 학습된 결과를 확인하는 코드
```python
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, aixs=1)
print('예측값 : ', sess.run(prediction, feed_dict={X:x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y:y_data}))
```
예측값인 model을 바로 출력하면 [0.2 0.7 0.1]과 같은 확률로 나오기 때문에, argmax함수를 사용하여 레이블 값을 출력한다.
<br/>
즉, 다음처럼 원-핫 인코딩을 거꾸로 한 결과를 만들어준다.
```python
[[0 1 0][1 0 0]] -> [1 0]
[[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
```
이제 간단하게 정확도 출력
```python
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```
전체 학습 데이터에 대한 예측값과 실측값을 tf.equal함수로 비교한 뒤, true/false 값으로 나온 결과를 다시 tf.cast 함수를 이용해 0과 1로 바꾸어 평균을 내면 간단히 정확도를 구할 수 있다.
<br/><br/>
손실값이 점점 줄어드는 것을 확인할 수 있다.

하지만, 학습 횟수를 아무리 늘려도 정확도가 크게 높아지지 않는다.<br/>
그 이유은 신경망이 한 층 밖에 안 되기 때문인데, 층 하나만 더 늘리면 쉽게 해결 가능하다.

## 4.3 심층 신경망 구현하기

> 신경망의 층을 둘 이상으로 구성한 심층 신경, 즉, 딥러닝을 구현해보자

앞서 만들었던 신경망 모델에 가중치와 편향을 추가하면 된다.

```python
W1 = tf.Variable(tf.random_uniform([2,10], -1, 1))
W2 = tf.Variable(tf.random_uniform([10, 3], -1, 1))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))
```
위 코드의 의미
```python
#가중치
W1 = [2, 10] -> [특징, 은닉층의 뉴런 수 ]
W2 = [10, 3] -> [은닉층의 뉴런 수, 분류 수]

#편향
b1 = [10] -> 은닉층의 뉴런 수
b2 = [3] -> 분류 수
```
- 입력층 : 특징
- 출력층 : 분류 개수
- 중간의 연결 부분 : 맞닿은 층의 뉴런 수
    - 중간의 연결부분을 **은닉층**이라 하며, 은닉층의 뉴런 수는 하이퍼파라미터이니 실험을 통해 가장 적절한 수를 정하면 된다.

<br/>
특징 입력값 <- 첫 번째 가중치, 편향, 활성화 함수

```python
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)
```
- 출력층
    - 두 번째 가중치와 편향을 적용하여 최종 모델을 만든다.
    - 은닉층에 두 번째 가중치 W2[10, 3]와 편향 b2[3]을 적용하면 3개의 출력값을 가진다
    
```python
model = tf.add(tf.matmul(L1, W2), b2)
```

> 앞 절에서 본 기본 신경망 모델에서는 출력층에 활성화 함수를 적용하였으나, 보통 출력층에 활성화함수를 사용하지 않는다.

- 손실함수 작성
    - 교차 엔트로피 함수 사용
        - 다만, 이번에는 텐서플로가 기본 제공하는 교차 엔트로피 함수를 이용

```python
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
```
- 최적화 함수 `AdamOptimizer`
    - 앞서 사용했던 `GradientDescentOptimizer`보다 보편적으로 성능이 좋다.
    - 하지만 모두 좋은 것은 아님

- 앞과 같이 쓰는 코드
    - 학습 진행
    - 손실값
    - 정확도 측정
    