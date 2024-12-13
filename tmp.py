from torch_geometric_temporal.nn.recurrent import ConvLSTM

# 모델 정의
conv_lstm = ConvLSTM(
    in_channels=1,   # 입력 채널 수 (e.g., 흑백 이미지 = 1)
    out_channels=64,  # ConvLSTM이 생성하는 출력 채널 수
    kernel_size=(3, 3),  # 커널 크기
    num_layers=3,  # ConvLSTM의 레이어 수
    bias=True,  # Bias 사용 여부
    batch_first=True  # 배치가 첫 번째 차원인지 여부
)

# 예제 입력 (배치 크기, 시간 단계, 채널, 높이, 너비)
input_tensor = torch.rand(8, 5, 1, 64, 64)  # (Batch, Time, Channels, Height, Width)
output, (h_n, c_n) = conv_lstm(input_tensor)  # Forward Pass

print("Output Shape:", output.shape)  # (Batch, Time, Channels, Height, Width)
