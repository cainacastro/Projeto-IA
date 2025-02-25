# Deep Q-Network (DQN) Training

Este repositório contém um script para treinar um agente de aprendizado por reforço usando DQN (Deep Q-Network) no ambiente da biblioteca Gymnasium.

## Requisitos
Antes de executar o script, instale as bibliotecas necessárias:

### 1. Criar e ativar um ambiente virtual (opcional, mas recomendado)
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate
```

### 2. Instalar dependências
```bash
pip install numpy torch torchvision matplotlib pillow

# Instalar o Gym com suporte ao Atari
pip install gym[atari]

# Instalar o PyTorch com suporte a GPU
pip install torch torchvision

# Instalar o OpenCV para processamento de frames
pip install opencv-python-headless

# Instalar ale-py (para rodar jogos do Atari)
pip install ale-py

# Atualizar ale-py e gymnasium
pip install --upgrade ale-py
pip install --upgrade gymnasium
```

Caso precise instalar pacotes adicionais, consulte a documentação do Gymnasium e PyTorch.

## Como Rodar o Script
1. Certifique-se de que todas as dependências estão instaladas.
2. Execute o script Python:
 ```bash
  # Instalar o Gym com suporte ao Atari
  pip install gym[atari]

  # Instalar o PyTorch com suporte a GPU
  pip install torch torchvision
  
  # Instalar o OpenCV para processamento de frames
  pip install opencv-python-headless
  
  # Instalar ale-py (para rodar jogos do Atari)
  pip install ale-py

  # Atualizar ale-py e gymnasium
  pip install --upgrade ale-py
  pip install --upgrade gymnasium
 ```

O script iniciará o treinamento do agente e salvará o modelo periodicamente.

## Salvamento e Carregamento do Modelo
O modelo é salvo periodicamente no diretório `MODEL_DIR`. Para carregar um modelo salvo, basta garantir que o arquivo existe antes de iniciar o treinamento.

