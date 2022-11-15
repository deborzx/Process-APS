#open cv carrega a imagem pela web cam, reconhece
#media pipe faz o reconhecimento
import cv2
import mediapipe as mp

#conectando na web cam
WebCam = cv2.VideoCapture(0)

#usando solução do mediapipe
solution_recog_face = mp.solutions.face_detection
#vai pegar a imagem e reconhecer o rosto
recog_faces = solution_recog_face.FaceDetection()
#cria o desenho pelo rosto
contorno = mp.solutions.drawing_utils


#frame: videos sao varias fotos que a cada segundo passam pela nossa vista, entao para nao ficar congelado
#o reconhecimento, é criado esse while
while True:
    #le infos da web cam
    verificador, frame = WebCam.read()
    
    if not verificador:
        break

    #reconhecer rostos dentro
    lista_rostos = recog_faces.process(frame)

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            #desenha rosto na imagem
            contorno.draw_detection(frame, rosto)
    cv2.imshow("TESTE", frame)


    # quando apertar ESC ele para o loop // waitKey - milisegundos
    if cv2.waitKey(10) == 27:
        break




WebCam.release()