from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# torchviz 라이브러리는 PyTorch 모델의 실행 그래프와 연산 과정을 시각화하는데 사용되는 라이브러리
# 내부적으로 Graphviz를 활용하여 모델의 연산 그래프를 이해하기 쉽게 그림으로 표현한다.
# - make_dot 함수:
# PyTorch 모델의 실행 그래프를 생성한다.
# 이 함수는 모델의 출력값과 파라미터를 입력으로 받아 그래프를 생성하며, 이를 이미지 파일로 저장할 수도 있다.

# torch.nn 모듈
# PyTorch 에서 신경망 모델을 구축하기 위한 다양한 클래스와 함수를 제공하는 핵심 모듈이다.
# 이 모듈은 딥러닝 모델의 레이어, 손실 함수, 활성화 함수 등을 정의하고 관리하는데 사용된다.
# nn.Module -> 모든 신경망 모델의 기본 클래스, 사용자 정의 모델을 만들 때 상속받아 사용하며, 레이어 정의와 순전파(forward) 과정을 구현한다.
# nn.Linear: 선형 변환(fully connected layer)을 수행하는 클래스
# nn.Identity: 입력값을 그대로 출력하는 레이어, 특정 연산을 생략하거나 모듈 대체 시 사용

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# 덧셈
z = x + y
print(z)

make_dot(z, params={x: x, y: y, z:z}, show_attrs=True, show_saved=True)

# s = z.sum()
#
# make_dot(s, params={x: x, y: y, z: z, s: s}, show_attrs=True, show_saved=True)