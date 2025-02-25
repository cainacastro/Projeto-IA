import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import torchvision.transforms as T
from PIL import Image
from collections import deque
from config import *
from cnn import CNN
from replay_buffer import ReplayBuffer


transform = T.Compose([
    T.Grayscale(),
    T.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
    T.ToTensor(),
    T.Normalize(0, 1)
])

gym.register_envs(ale_py)

env = gym.make(ENV_NAME, frameskip=FRAME_SKIP)

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    frame = transform(frame)
    return frame

def select_action(state, epsilon, model, action_size):
    if random.random() < epsilon:
        action = random.randint(0, action_size - 1)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q_values = model(state)
            action = torch.argmax(q_values).item()

    return action

action_size = env.action_space.n
policy_net = CNN(action_size).to(DEVICE)
target_net = CNN(action_size).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_filename = os.path.join(MODEL_DIR, MODEL_FILENAME)
try:
    policy_net.load_state_dict(torch.load(model_filename))
except FileNotFoundError:
    print(f"Modelo {model_filename} não encontrado. Iniciando treinamento do zero.")

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

reward_medio = 0
pontuacao_media = 0
highest_pontuacao = 0
epoca = 1
rewards = []
epsilons = []

for episode in range(NUM_EPISODES):
    frame_stack = deque(maxlen=FRAME_STACK) 
    state = preprocess_frame(env.reset()[0]).squeeze(0) 
    prev_lives = env.unwrapped.ale.lives()
    for _ in range(FRAME_STACK):
        frame_stack.append(state)
        
    stacked_state = np.stack(frame_stack, axis=0)
    total_reward = 0
    pontuacao = 0

    for t in range(PASSOS_POR_EPISODIO): 
        action = select_action(stacked_state, EPSILON, policy_net, action_size)
        next_frame, reward, terminated, truncated, info = env.step(action)

        reward *= MULTIPLICAOR_RECOMPENSA
        if t > PASSOS_MULT_RECOMPENSA:
          reward *= MULTIPLICAOR_RECOMPENSA
        if reward > 0:
          pontuacao += 1
        if 'lives' in info:
            current_lives = info['lives']
        if current_lives < prev_lives:
            reward += PENALIDADE_LOST_LIFE 

        prev_lives = current_lives

        next_state = preprocess_frame(next_frame)
        frame_stack.append(next_state.squeeze(0))
        stacked_next_state = np.stack(frame_stack, axis=0)

        replay_buffer.push(stacked_state, action, reward, stacked_next_state, terminated)

        stacked_state = stacked_next_state
        state = next_state
        total_reward += reward

        if len(replay_buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            q_values = policy_net(states.to(DEVICE)).gather(1, actions.unsqueeze(1).to(DEVICE)).squeeze(1)
            next_q_values = target_net(next_states.to(DEVICE)).max(1)[0].detach()
            expected_q_values = rewards.to(DEVICE) + (1 - dones.to(DEVICE)) * GAMMA * next_q_values

            loss = nn.SmoothL1Loss()(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if terminated or truncated:
            print(f"LEVOU {t} PASSOS ")
            break

    EPSILON = max(EPSILON_MIN, EPSILON - EPSILON_DECAY)

    if pontuacao > highest_pontuacao:
      highest_pontuacao = pontuacao

    pontuacao_media += pontuacao
    pontuacao_media /= 2

    print(f"Episódio {episode + 1}, Epsilon: {EPSILON:.3f}, Pontuacao Media : {pontuacao_media:.1f}, Maior Pontuacao : {highest_pontuacao}")

    if (episode + 1) % NUM_EPISODES_SAVE == 0:
        model_path = f'{MODEL_DIR}/dqn_model_{episode + 1}.pth'
        torch.save(policy_net.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")

env.close()
