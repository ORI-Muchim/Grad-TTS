import torch

# 저장된 모델의 상태 딕셔너리 불러오기
state_dict = torch.load('./models/kss/G_1000.pth')

# 모든 파라미터의 데이터 타입 출력
for name, param in state_dict.items():
    print(f'{name}: {param.dtype}')
