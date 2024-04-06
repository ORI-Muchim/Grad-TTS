import torch

# 모델의 상태 딕셔너리 불러오기
state_dict = torch.load('./logs/kss_test/grad_36.pt')

# 'model_state' 항목에서 모델 파라미터 추출하기
model_state = state_dict.get('model_state', {})

# 모델 상태 딕셔너리 내부를 순회하며 모든 파라미터 정보 출력
for name, param in model_state.items():
    # 파라미터가 텐서인지 확인
    if torch.is_tensor(param):
        print(f'{name}: {param.dtype}')
    else:
        # 파라미터가 텐서가 아닌 경우, 그 타입을 출력
        print(f'{name} is not a tensor, but a {type(param)}')
