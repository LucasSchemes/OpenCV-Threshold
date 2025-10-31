# 🔵 Detecção de Círculos e Cálculo de Área de Triângulo com OpenCV  

Este projeto utiliza **Python + OpenCV** para capturar imagens da **webcam**, detectar círculos pretos em uma folha, formar um triângulo com três deles e calcular sua **área** em tempo real. Além disso, o código aplica técnicas de **pré-processamento de imagem** para melhorar a detecção e exibe as informações na tela.  

---

## 🚀 Funcionalidades
- Captura de vídeo em tempo real da webcam.  
- Pré-processamento com:
  - Ajuste de brilho e contraste.  
  - Conversão para escala de cinza.  
  - CLAHE (Equalização Adaptativa de Histograma).  
  - Desfoque Gaussiano.  
  - Limiarização de Otsu.  
- Detecção de círculos com **HoughCircles**.  
- Exibição de:
  - Posição (x, y) e raio de cada círculo.  
  - Triângulo formado pelos três primeiros círculos detectados.  
  - Área do triângulo, calculada pela fórmula de **Heron**.  
  - Centroide do triângulo.  
- Suavização das posições dos círculos usando histórico de detecções para reduzir tremores.  
- Encerramento do programa ao pressionar **Q**.  

---

## 🖼️ Demonstração
O programa exibe duas janelas:  
- **Imagem Binária (Otsu):** resultado da segmentação em preto e branco.  
- **Círculos e Triângulo:** visualização dos círculos detectados, suas coordenadas e o triângulo formado.  

---

## 📦 Requisitos
- Python 3.8+  
- [OpenCV](https://opencv.org/)  
- [NumPy](https://numpy.org/)  

---

## 🔧 Instalação
Clone este repositório e instale as dependências:

```bash
git clone https://github.com/LucasSchemes/DeteccaoCirculos-OpenCV.git
cd DeteccaoCirculos-OpenCV
pip install opencv-python numpy
```

## Execução

Basta rodar o script principal

```bash
python cam.py

