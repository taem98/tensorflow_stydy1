#chapter3 텐서플로 프로그래밍 101


> 텐서플로는 딥러닝용으로만 사용하는 것이 아니다. 그래프 형태의 수학식 계산을 수행하는 핵심 라이브러리를 구현한 후, 그 위에 딥러닝을 포함한 여러 머신러닝을 쉽게 수행하는 핵심 라이브러리

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
```

