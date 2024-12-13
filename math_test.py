import torch
import torch.nn as nn
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드 및 전처리
data = pd.read_csv('7_I_8640.csv')
data['DATETIME'] = pd.to_datetime(data['DATETIME'])
data = data.sort_values(by='DATETIME')

columns_of_interest = ['10m_rainfall', '1hr_rainfall', '3hr_rainfall', 
                       'gutter_level', 'river_level', 'danger_level']
data = data[columns_of_interest].dropna()

scaler = MinMaxScaler()
data[columns_of_interest] = scaler.fit_transform(data[columns_of_interest])

# 시계열 데이터 생성
time_steps = 10
X, y = [], []
for i in range(len(data) - time_steps):
    X.append(data.iloc[i:i + time_steps].values)
    y.append(data.iloc[i + time_steps].values)

X, y = np.array(X), np.array(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 정의
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력만 사용
        return out

# 모델 설정 로드
with open('data/math.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_dim = config['input_dim']
hidden_dim = config['hidden_dim']
output_dim = config['output_dim']

model = ConvLSTM(input_dim, hidden_dim, output_dim)

# 미리 학습된 가중치 로드
model.load_state_dict(torch.load('math_custom.pt'))

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
model.train()
for epoch in range(10):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# 예측
model.eval()
with torch.no_grad():
    predictions = model(X[-1:])
    print("1분, 10분, 30분, 1시간 뒤 예측 결과:", predictions.numpy())
