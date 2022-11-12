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

varGlobal1 = img01
varGlobal2 = img04

#para colocar a imagem com a cor padrão
varGlobal1 = cv2.cvtColor(varGlobal1, cv2.COLOR_BGR2RGB)
varGlobal2 = cv2.cvtColor(varGlobal2, cv2.COLOR_BGR2RGB)


#printando as coordenadas da imagem
LocalizationFace = fr.face_locations(varGlobal1)[0]


#calculando area retangular da imagem
cv2.rectangle(varGlobal1, (LocalizationFace[3], LocalizationFace[0]), (LocalizationFace[1], LocalizationFace[2]), (0,255,0),2)


#para retornar as medidas do rosto da pessoa no terminal
medidasPic = fr.face_encodings(varGlobal1)[0]
medidasPicTest = fr.face_encodings(varGlobal2)[0]

#comparando medidas
TestComparation = fr.compare_faces([medidasPic], medidasPicTest)

print(TestComparation)

if TestComparation == [True]:
    print("É a mesma pessoa!")

else:
    print("Não é a mesma pessoa!")


#mostra a imagem
cv2.imshow('Imagem Original', varGlobal1)
cv2.imshow('Imagem de Teste', varGlobal2)
cv2.waitKey(0)


