import cv2 as cv
import numpy as np
import math

def calcular_area_triangulo(p1, p2, p3):
    def distancia(p1, p2): 
        if np.isfinite(p1[0]) and np.isfinite(p1[1]) and np.isfinite(p2[0]) and np.isfinite(p2[1]): 
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        else:
            return 0

    a = distancia(p1, p2)
    b = distancia(p2, p3)
    c = distancia(p3, p1)

    # Evitar divisão por zero
    if a == 0 or b == 0 or c == 0:
        return 0

    s = (a + b + c) / 2
    area_expression = s * (s - a) * (s - b) * (s - c)

    if area_expression > 0 and np.isfinite(area_expression):
        return math.sqrt(area_expression)
    else:
        return 0

def calcular_centroide(p1, p2, p3): 
    x_centroid = (p1[0] + p2[0] + p3[0]) // 3
    y_centroid = (p1[1] + p2[1] + p3[1]) // 3
    return (x_centroid, y_centroid)

def ordenar_circulos(centros): #angulo para girar a imagem
    centros = sorted(centros, key=lambda p: math.atan2(p[1], p[0]))
    return centros

def validar_circulos(centros): # tirar circulos irreais (distancia muito perto)
    
    if len(centros) >= 3:
        d1 = math.dist(centros[0], centros[1])
        d2 = math.dist(centros[1], centros[2])
        d3 = math.dist(centros[2], centros[0])
        # Evitar triângulos muito pequenos ou degenerados
        if d1 < 20 or d2 < 20 or d3 < 20:
            return False
    return True

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()


centros_historico = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro: Não foi possível capturar a imagem.")
        break

    alpha = 1.8
    beta = -70
    enhanced_frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

    gray_img = cv.cvtColor(enhanced_frame, cv.COLOR_BGR2GRAY) 
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_img = clahe.apply(gray_img)

    blurred_img = cv.GaussianBlur(gray_img, (9, 9), 2) 

    ret, binary_img = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    circulos = cv.HoughCircles(binary_img, 
                                cv.HOUGH_GRADIENT, dp=2, minDist=80,
                                param1=100, param2=65, minRadius=30, maxRadius=85)

    centros = []

    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        
        for circulo in circulos[0, :]:
            x, y, raio = circulo[0], circulo[1], circulo[2]
            centros.append((x, y)) 

            cv.circle(frame, (x, y), 2, (0, 0, 255), 3)
            cv.circle(frame, (x, y), raio, (0, 255, 0), 2)

            coords_text = f"({x}, {y})" 
            cv.putText(frame, coords_text, (x + 10, y), cv.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2, cv.LINE_AA)

        if len(centros) >= 3 and validar_circulos(centros):
            centros = ordenar_circulos(centros)

            # Adicionar a posição dos centros ao histórico e calcular média
            centros_historico.append(centros[:3]) 
            if len(centros_historico) > 5:
                centros_historico.pop(0) #remove conjunto mais antigo
            centros_media = np.mean(centros_historico, axis=0) # média que suaviza variações.

            # Converter de float para int
            centros_media = np.int32(centros_media)

            # Desenhar o triângulo
            cv.line(frame, tuple(centros_media[0]), tuple(centros_media[1]), (255, 0, 0), 2)
            cv.line(frame, tuple(centros_media[1]), tuple(centros_media[2]), (255, 0, 0), 2)
            cv.line(frame, tuple(centros_media[2]), tuple(centros_media[0]), (255, 0, 0), 2)

            area = calcular_area_triangulo(centros_media[0], centros_media[1], centros_media[2])
            centroide = calcular_centroide(centros_media[0], centros_media[1], centros_media[2])

            height, width, _ = frame.shape
            area_text = f"Area: {area:.2f}"
            cv.putText(frame, area_text, (centroide[0], centroide[1]), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

    cv.imshow('Imagem Binaria (Otsu)', binary_img)
    cv.imshow('Circulo e triangulo', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
