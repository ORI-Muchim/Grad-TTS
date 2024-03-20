import torch
from model import GradTTS
from text.symbols import symbols
from torch.quantization import get_default_qconfig, QConfig, prepare, convert, default_observer, default_weight_observer

import params

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
text_cleaners = params.text_cleaners
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

# ConvTranspose{n}d 계층에 대한 QConfig 생성
# 옵저버 클래스를 직접 전달하고, with_args() 메서드를 사용하여 필요한 경우 인자를 전달합니다.
conv_transpose_qconfig = QConfig(
    activation=default_observer.with_args(),  # 활성화에 대해서는 기본 관찰자 클래스 사용
    weight=default_weight_observer.with_args())  # 가중치에 대해 전체 가중치 관찰자 클래스 사용

def quantize_model(model_path, quantized_model_path, device='cpu'):
    model = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    # 모델의 QConfig 설정
    model.qconfig = get_default_qconfig('qnnpack')

    # 특정 계층의 QConfig를 None으로 설정하여 양자화 제외
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Embedding):
            module.qconfig = None  # Conv1d와 Embedding 계층 양자화 제외

    # 양자화 준비 및 적용
    prepare(model, inplace=True)
    convert(model, inplace=True)

    # 양자화된 모델과 추가 정보를 포함한 새로운 체크포인트 생성
    quantized_checkpoint = {
        'model': model.state_dict(),
        'iteration': checkpoint.get('iteration', None),
        'optimizer': checkpoint.get('optimizer', None),
        'learning_rate': checkpoint.get('learning_rate', None)
    }

    # 양자화된 모델 저장
    torch.save(quantized_checkpoint, quantized_model_path)
    print(f"양자화된 모델이 {quantized_model_path}에 저장되었습니다.")

# 학습된 모델의 가중치가 저장된 파일 경로
MODEL_WEIGHT_PATH = './logs/kss/grad_10.pth'
quantized_model_path = 'quantized_model.pth'
quantize_model(MODEL_WEIGHT_PATH, quantized_model_path, device='cpu')
