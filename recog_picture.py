import cv2
import face_recognition as fr

#imagens

img01 = fr.load_image_file('./img-faces/pic1.jpg')
img02 = fr.load_image_file('./img-faces/pic2.jpg')
img03 = fr.load_image_file('./img-faces/pic3.jpeg')
img04 = fr.load_image_file('./img-faces/pic4.jpeg')
img05 = fr.load_image_file('./img-faces/pic5.jpg')
img06 = fr.load_image_file('./img-faces/pic6.jpg')
img07 = fr.load_image_file('./img-faces/pic7.jpg')
img08 = fr.load_image_file('./img-faces/pic8.jpg')


varGlobal = img03 

#para colocar a imagem com a cor padrão
varGlobal = cv2.cvtColor(varGlobal, cv2.COLOR_BGR2RGB)


#printando as coordenadas da imagem
LocalizationFace = fr.face_locations(varGlobal)[0]

#calculando area retangular da imagem
cv2.rectangle(varGlobal, (LocalizationFace[3], LocalizationFace[0]), (LocalizationFace[1], LocalizationFace[2]), (0,255,0),2)


#para retornar as medidas do rosto da pessoa no terminal
medidasPic = fr.face_encodings(varGlobal)[0]


#mostra a imagem
cv2.imshow('Imagem Original', varGlobal)

cv2.waitKey(0)


