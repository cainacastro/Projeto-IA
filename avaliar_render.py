import gymnasium as gym
import ale_py
import cv2
import numpy as np
import torch
import torch.nn as nn
import random
import torchvision.transforms as T
from PIL import Image
from collections import deque
from config import *

# Definindo a transformação para o pré-processamento dos frames
transform = T.Compose([
    T.Grayscale(),  
    T.Resize((FRAME_HEIGHT, FRAME_WIDTH)),  
    T.ToTensor(),  
    T.Normalize(0, 1)  # Mantém os valores próximos de zero
])

gym.register_envs(ale_py)

# Criar o ambiente do Gymnasium
env = gym.make(ENV_NAME, frameskip=FRAME_SKIP)

# Função para processar frames
def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    frame = transform(frame)
    return frame.squeeze(0)

class CNN(nn.Module):
    def __init__(self, action_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=CONV1_KERNEL_SIZE, stride=CONV1_STRIDE)
        self.conv2 = torch.nn.Conv2d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=CONV2_KERNEL_SIZE, stride=CONV2_STRIDE)
        self.conv3 = torch.nn.Conv2d(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=CONV3_KERNEL_SIZE, stride=CONV3_STRIDE)
        self.fc1 = torch.nn.Linear(FC1_UNITS_IN, FC1_UNITS_OUT)
        self.fc2 = torch.nn.Linear(FC1_UNITS_OUT, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Retorna os Q-values para cada ação

model = CNN(ACTION_SIZE).to(DEVICE)

# Carregar o modelo salvo
model.load_state_dict(torch.load('test_ricas/logs_krl/dqn_model_1560.pth', map_location=DEVICE))
model.eval()

# Função para escolher a ação com base na política treinada
def select_action(state, model):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0).squeeze(1)  # Remove dimensão extra
        q_values = model(state)
        return torch.argmax(q_values).item()  # Melhor ação

# Criar o ambiente
env = gym.make(ENV_NAME, render_mode='human')
obs, info = env.reset()

frame_stack = deque(maxlen=FRAME_STACK)

# Inicializa a pilha com 4 frames iguais
first_frame = preprocess_frame(obs)
for _ in range(FRAME_STACK):
    frame_stack.append(first_frame)

# Criar estado inicial empilhado corretamente
stacked_state = np.stack(frame_stack, axis=0)
stacked_state = np.expand_dims(stacked_state, axis=0)  # Adicionar dimensão de batch

# Rodar o ambiente e mostrar a interface
done = False
total_reward = 0

while not done:
    action = select_action(stacked_state, model)  # Seleciona a ação usando a DQN
    
    next_obs, reward, terminated, truncated, info = env.step(action)  # Executa a ação
    next_frame = preprocess_frame(next_obs)
    
    # Adicionar novo frame na pilha
    frame_stack.append(next_frame)
    stacked_state = np.stack(frame_stack, axis=0)  # Atualiza o estado empilhado
    stacked_state = np.expand_dims(stacked_state, axis=0)  # Adicionar dimensão de batch
    total_reward += reward
    
    # Renderizar o ambiente
    env.render()
    
    if terminated or truncated:
        break

# Exibe a recompensa total obtida
print(f"Recompensa Total: {total_reward}")

env.close()
