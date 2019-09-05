# chapter3 텐서플로 프로그래밍 101


> 텐서플로는 딥러닝용으로만 사용하는 것이 아니다. 그래프 형태의 수학식 계산을 수행하는 핵심 라이브러리를 구현한 후, 그 위에 딥러닝을 포함한 여러 머신러닝을 쉽게 수행하는 핵심 라이브러리이다.

### 이번 장에서 배울 것들
- 텐서 tensor
- 플레이스 홀더 placeholder
- 변수 variable
- 연산의 개념
- 기본적인 그래프 실행 방법


## 3.1 텐서와 그래프 실행

### 텐서 플로우 임포트
```python
import tensorflow as tf
```

### tf.constant로 상수를 hello 변수에 저장
```python
hello = tf.constant('hello, TensorFlow!')
print(hello)

#결과 : Tensor("Const:0", shape=(), dtype=string)
#자료형 : 텐서
#상수를 담고 있음

```
> 텐서 : 텐서플로의 자료형, 랭크(Rank)와 셰이프(Shape)의 개념을 가지고 있다.

```python
3 # 랭크 0, 셰이프 []
[1., 2., 3.] # 랭크 1, 셰이프[3]
[[1.,2.,3.],[4.,5.,6.]] # 랭크 2 셰이프[2,3]
[[[1.,2.,3.]],[[4.,5.,6.]]] # 랭크 3 셰이프 [2,1,3]
```

- 텐서의 자료형 : 배열과 비슷
    - 랭크 : 차원의 수
        - 0 : 스칼라
        - 1 : 벡터
        - 2 : 행렬
        - 3이상 : n-Tensor , n차원 텐서
    - 셰이프 : 각 차원의 요소 개수, 텐서의 구조를 설명해준다<br/><br/>


- dtype : 텐서에 담긴 자료형
    - string, float, int 등<br/><br/>

### 다양한 연산

```python
#이러한 텐서로 다양한 연산 가능
 
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)


#결과  : Tensor("Add:0", shape=(), dtype=int32)
```

### 왜 42가 안 나올까?

> 텐서플로 프로그램의 구조가 2가지로 분리되어 있기 때문

1. 그래프 생성
2. 그래프 실행

```python
# ex>

# 그래프 생성
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)

# 그래프 실행
sess.run(c)
```

- 그래프 : 텐서들의 연산 모음
- 지연실행(lazy evaluation) : 텐서와 텐서의 연산들을 먼저 정의하여 그래프를 만들고, 이후 필요할 때 연산을 실행하는 코드를 넣어서 원하는 시점에 실제 연산을 수행하도록 함

### 실제 수행은?

> 그래프의 실행은 Session 안에서 이뤄져야 한다. 

Session 객체와 run 메서드

```python
sess = tf.Session()

print(sess.run([a,b,c]))

sess.close()

# [10, 32, 42]
```

## 3.2 플레이스홀더와 변수

> 텐서플로로 프로그래밍을 할 때 가장 중요한 두가지. **플레이스홀더**, **변수**

### 플레이스 홀더 ? 변수?

- 플레이스 홀더 : 매개변수
    - 그래프에 사용할 입력값을 나중에 받기 위해 사용
- 변수 : 그래프를 최적화하는 용도

### 플레이스 홀더
> 사용법
```python
# None은 크기가 정해지지 않았음을 의미한다.

X = tf.placeholder(tf.float32, [None, 3])
print(X)

#Tensor("Placeholder:0", shape=(?, 3), dtype=float32)
```

나중에 플레이스 홀더 X에 넣을 자료를 다음과 같이 정의할 수 있다.<br/>
앞서 텐서 모양을 (?,3)으로 정의했으므로, 두 번째 차원은 요소를 3개씩 가지고 있어야 한다.

```python 
x_data = [[1,2,3],[4,5,6]]
```

다음은 변수들 정의<br/>

```python

# 각각 W와 b에 텐서플로의 변수를 생성하여 할당한다.
# random_normal 함수 : 정규분포의 무작위 값으로 초기화

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

# 물론 원하는 값으로 초기화도 가능
#W = tf.Variable(tf.random_normal([[0.1, 0.1],[0.2, 0.2],[0.3, 0.3]]))

```
다음으로 입력값과 변수들을 계산할 수식 작성<br/>
X와 W가 행렬이기 때문에 **tf.matmul**함수를 사용해야 한다.<br/>
- 행렬이 아닐 경우 곱셈연산자 **(*)**나 **tf.mul** 함수를 사용

```python
expr = tf.matmul(X,W) + b
```

X 를 [2,3] 를 정의 했기 때문에 W를 [3,2]로 정의<br/><br/>

이제 결과를 출력하자

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("======= x_data =======")
print(x_data)
print("======= W =======")
print(sess.run(W))
print("======= b =======")
print(sess.run(b))
print("======= expr =======")
print(sess.run(expr, feed_dict={X:x_data}))

sess.close()
```
- tf.global_variables_initializer()
    - 앞에서 정의한 변수들을 초기화
    - 기존에 학습한 값들을 가져와서 사용하는 것이 아닌 **처음 실행하는 것**이라면, 연산 실행 전에 꼭 사용해서 변수들을 초기화 해야함
<br/>
- feed_dict
    - 그래프를 실행할 때 사용할 입력값 지정
<br/>
expr수식에는 X, W, b를 사용했는데 이 중에서 X가 플레이스홀더라서 X에 값을 넣어주지 않으면 계산에 사용할 값이 없으므로 에러가 난다.
<br/>
따라서 미리 정의해둔 x_data를 X의 값으로 넣어주었다.
<br/>
<br/>
```python
# 결과
======= x_data =======
[[1, 2, 3], [4, 5, 6]]
======= W =======
[[-1.1200693   0.72515726]
 [-1.0583055   1.0693772 ]
 [ 0.70268965  0.02659263]]
======= b =======
[[ 0.16556801]
 [-1.9831709 ]]
======= expr =======
[[-0.96304333  3.1092577 ]
 [-7.538838    6.423899  ]]
```


## 3.3 선형 회귀 모델 구현하기

### 선형회귀란?

> 선형회귀 : 주어진 x와 y값을 가지고 서로 간의 관계를 파악하는 것

이 관계를 알면 x에 대한 y값을 구할 수 있고, 이것이 머신러닝의 기본


### 선형회귀 모델을 만들고 실행해보자
> X 와 Y의 상관관계를 분석하는 기초적인 선형회귀 모델을 만들어보자

x_data와 y_data의 상관관계 파악하기
```python
x_data = [1,2,3]
y_data = [1,2,3]
```

먼저, x와 y의 상관관계를 설명하기 위한 변수들인 W와 b를 각각 -1.0부터 1.0 사이의 **균등분포**를 가진 무작위 값으로 초기화

```python 

W = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
b = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
```
다음으로 자료를 입력 받을 플레이스홀더를 설정합니다.

```python
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
```
그 다음으로 X와 Y의 상관관계를 분석하기 위한 수식을 작성한다.

```python
hypothesis = W * X + b

# X 가 주어졌을 때 Y를 만들어 낼 수 있는 W 와 b를 찾겠다.
# 선형 회귀는 물론 신경망 학습에 가장 기본이 되는 수식
```
- w : 가중치
- b : 편향

<br/>

다음으로 손실 함수를 작성

### 손실함수란? 
> 한 쌍(x,y)의 데이터에 대한 **손실값**을 계산하는 함수

손실값이란?<br/>
> 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값
- 즉, 손실값이 작을 수록 그 모델이 X와 Y의 관계를 잘 설명함
    - '예측값과 실제값의 거리'를 가장 많이 사용
        - 우리도 이걸 사용 할 것
        - 예측값에서 실제값을 뺀 뒤 제곱
### 이 손실을 전체 데이터에서 구한 경우 이를 **비용**이라고 한다.

즉 **학습**이란 변수들의 값을 다양하게 넣어 계산해보면서 이 손실값을 최소화하는 W와 b의 값을 구하는 것이다.
<br/><br/>
비용 : 모든 데이터에 대한 평균값을 내어 구한다.

```python
cost = tf.reduce_mean(tf.square(hypothesis -Y))
```

[다양한 텐서플로 함수들](https://www.tensorflow.org/api_docs/)

<br/>
마지막으로 텐서플로가 기본 제공하는 **경사하강법**최적화 함수를 이용해 손실을 최소화 하는 연산 그래프를 생성

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)
```
- 최적화 함수
    - 가중치과 편향 값을 변경해가면서 손실값을 최소화하는 가장 최적화된 가중치과 편향 값을 찾아주는 함수
        - 이때 값들을 무작위로 변경하면 시간도 오래 걸리고 학습시간 예측도 어려움
        - 따라서 빠르게 최적화하기 위한, 빠르게 학습하기 위한 다양한 방법을 사용
        - **경사하강법**이 그 중 하나

### 경사하강법이란?
 > 함수의 기울기를 구하고 기울기가 낮은 쪽으로 계속 이동시키면서 최적의 값을 찾아나가는 방법

### learning_rate
> 최적화 함수의 매개변수, 학습률

학습을 얼마나 급하게 할 것인가?
- 값이 너무 크면
    - 최적의 손실값을 찾지 못하고 지나침
- 값이 너무 작으면
    - 학습 속도가 너무 느려짐

<br/>
이렇게 학습에 영향을 주는 변수를 **하이퍼파라미터**라고 한다.<br/>
이 값에 따라 학습 속도나 신경망 성능이 크게 달라질 수 있다.<br/>
머신러닝에서는 하이퍼파라미터를 잘 튜닝하는 것이 큰 과제이기도 하다.

<br/>

이제 선형 회귀 모델을 다 만들었다.<br/>
그래프를 실행해 학습시키고 결과를 확인할 차례<br/>
<br/>
세션 생성, 변수 초기화 후 **with** 기능을 써서 세션 블록을 만들고 세션 종료를 자동으로 처리할 수 있도록 한다.

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
```

최적화를 수행하는 그래프인 `train_op`를 실행하고, 실행 시마다 변화하는 손실값을 출력하는 코드<br/>
- 학습횟수 : 100번<br/>

```python
for step in range(100):
    -, cost_val = sess.run([train_op, cost], feed_dict = {X:x_data, Y:y_data})
    print(step, cost_val, sess.run(W), sess.run(b))
```





# + 행렬
> 텐서플로로 딥러닝을 하려면 행렬은 꼭 알아야 한다. 그 중에서도 **행렬곱**

### 행렬곱의 정의

- 행렬곱 A X B에 대해서 A의 열 수와 B의 행 수는 같아야 한다.
- 행렬곱 A X B를 계산한 AB의 크기는 A의 행 개수와 B의 열 개수가 된다.

# + 플레이스 홀더란 무엇인가?

[참고](https://gdyoon.tistory.com/5)

선언과 동시에 초기화 x <br/>
선언 후에 그 값을 전달한다.<br/>
따라서 반드시 실행 시 데이터가 제공되어야 한다.

**값을 전달한다?**
- 데이터를 상수값을 전달함과 같이 할당하는 것이 아니라 다른 텐서(Tensor)를 placeholder에 맵핑 시키는 것
<br/><br/>

placeholder의 전달 파라미터
```python
placeholder(
    dtype,
    shape=None,
    name=None
)
```
- dtype : 데이터 타입을 의미하며 반드시 적어주어야 한다.
- shape : 입력 데이터의 형태.
    - 상수, 다차원 배열 정보 등등
- name : 해당 placeholder의 이름을 부여. 
    - 필수는 아니다.

## 플레이스 홀더에 이름 부여하기

### name 매개변수

> 플레이스 홀더의 name 매개변수로는 플레이스 홀더의 이름을 설정한다.

- name 없이 설정한 placeholder를 출력하면 이름이 자동부여 된다.
    - ex. Placeholder, Placeholder_1

#### 이름을 쓰면 왜 좋을까?

- 텐서가 어떻게 사용되고 있는지 쉽게 알 수 있다.
- 텐서보드에서도 이름으로 출력해주기 때문에 디버깅 하기 쉽다
<br/>
    +변수와 연산 또는 연산함수에도 이름을 지정할 수 있다.

## 예제 1

```python
p_holder1 = tf.placeholder(dtype=tf.float32)
p_holder2 = tf.placeholder(dtype=tf.float32)
p_holder3 = tf.placeholder(dtype=tf.float32)
 
val1 = 5
val2 = 10
val3 = 3
 
ret_val = p_holder1 * p_holder2 + p_holder3
 
feed_dict = {p_holder1: val1, p_holder2: val2, p_holder3: val3}
result = sess.run(ret_val, feed_dict=feed_dict)
 
print(result)

# 53.0
```
- feed dictionaty (feed_dict)
    - placeholder에 다른 텐서를 할당하기 위해 활용함
    - 텐서 맵핑용
    - 위와 같이 feed_dict 변수를 할당해도 되고, 값을 대입시켜도 무방하다.

## 예제 2

> image Matrix(영상)의 정보와 각 라벨이 들어가 있다고 가정


```python
# 배열의 형태가 값으로 들어가도 무방

mat_img = [1,2,3,4,5]
label = [10, 20, 30, 40, 50]

ph_img = tf.placeholder(dtype=tf.float32)
ph_lb = tf.placeholder(dtype=tf.float32)

ret_tensor = ph_img + ph_lb

result = sess.run(ret_tensor, feed_dict = {ph_img:mat_img, ph_lb:label})
print(resurlt)

# [11,22,33,44,55]
```

