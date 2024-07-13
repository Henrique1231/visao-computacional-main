import cv2

def inicializar_detector_de_roupas():

    classificador_de_roupas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if classificador_de_roupas.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de roupas.")
    return classificador_de_roupas

def detectar_roupas(quadro, classificador_de_roupas):
    
    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
    roupas = classificador_de_roupas.detectMultiScale(cinza, scaleFactor=2.1, minNeighbors=10, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    return roupas

def desenhar_roupas(quadro, roupas):
    
    for (x, y, largura, altura) in roupas:
        cv2.rectangle(quadro, (x, y), (x + largura, y + altura), (245, 255, 0), 2) # BGR

def main():
    
    classificador_de_roupas = inicializar_detector_de_roupas()
    captura_de_video = cv2.VideoCapture(1)
    
    captura_de_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    captura_de_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not captura_de_video.isOpened():
        raise Exception("Não foi possível abrir a webcam.")
    
    print('\033[1;31;43m' + 'Buscando roupas...' + '\033[0;39;49m')
    
    try:
        while True:
            ret, quadro = captura_de_video.read()
            if not ret:
                break

            roupas = detectar_roupas(quadro, classificador_de_roupas)
            desenhar_roupas(quadro, roupas)

            cv2.imshow('Reconhecimento de roupas', quadro)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        captura_de_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
