import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from PIL import Image, ImageDraw

def sepia(image_path:str)->Image:
    img = Image.open(image_path)
    width, height = img.size

    pixels = img.load() #


    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255

            if tg > 255:
                tg = 255

            if tb > 255:
                tb = 255

            pixels[px, py] = (tr,tg,tb)

    return img


image = Image.open("Maple.jpg") # відкриття файлу зображення.
draw = ImageDraw.Draw(image) # створення інструменту для малювання
width = image.size[0] # визначення ширини картинки
height = image.size[1] # визначення висоти картинки
pix = image.load() # отримання значень пікселей для картинки

# зашумлення

factor = int(50)
print('------- триває перетворення --------------')
for i in range(width):
   for j in range(height):
      rand = random.randint(-factor, factor)
      a = pix[i, j][0] + rand   # додавання рандомного числа
      b = pix[i, j][1] + rand
      c = pix[i, j][2] + rand
      if (a < 0):
         a = 0
      if (b < 0):
         b = 0
      if (c < 0):
         c = 0
      if (a > 255):
         a = 255
      if (b > 255):
         b = 255
      if (c > 255):
         c = 255
      draw.point((i, j), (a, b, c))

image.save("Maple4.jpg", "JPEG")
sepia("Maple4.jpg").save("Maple4.jpg", "JPEG")
plt.title('Зображення після обробки зашумленням та сепією')
plt.imshow(Image.open("Maple4.jpg"))
plt.show()
# Гауссова фільтрація
img = cv2.imread('Maple4.jpg')
plt.title('Зображення, яке отримується через imread в cv2')
plt.imshow(img)
plt.show()
# виклик методу - GaussianBlur
blur4 = cv2.GaussianBlur(img,(5,5),0)
# відображення результату
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur4),plt.title('GaussianBlur')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('Maple4.jpg')
kernel = np.ones((5,5),np.float32)/25
# виклик методу 2D Convolution
blur1 = cv2.filter2D(img,-1,kernel)
# відображення результату
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur1),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# Розмиття через згладжування
img = cv2.imread('Maple4.jpg')
# виклик методу - blur
blur2 = cv2.blur(img,(5,5))
# відображення результату
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur2),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


# Медіанний фільтр
img = cv2.imread('Maple4.jpg')
# виклик методу -blur
blur4 = cv2.medianBlur(img,5)
# відображення результату
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur4),plt.title('Median Filtering')
plt.xticks([]), plt.yticks([])
plt.show()

# Двостороння фільтрація
img = cv2.imread('Maple4.jpg')
# виклик методу -blur
blur5 = cv2.bilateralFilter(img,9,75,75)
# відображення результату
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur5),plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()



# згладжування з видыленням контуру за алгоритмом kernel
img = blur5
#виклик методу 2D Convolution
blur1=cv2.Canny (img, 100,200)
# відображення результату
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur1),plt.title('Згладжування із виділенням контуру')
plt.xticks([]), plt.yticks([])
plt.show()

from pylab import *
# кольорова векторізація з внутрішнім заповненням
# зчитування піксельного зображення в масив
im = array(Image.open('Maple4.jpg').convert('L'))
# створити нову фігуру
figure()
# відобразити контур
title('кольорова векторізація з внутрішнім заповненням')
contour(im, origin='image')
axis('equal')
show()
# монохромна векторізація без внутрішнього заповнення
im = array(Image.open('Maple4.jpg').convert('L'))
# створити нову фігуру
figure()
# відобразити контур монохром без внутрішнього заповнення
title('монохромна векторізація без внутрішнього заповнення')
contour(im, levels=[245], colors='black', origin='image')
axis('equal')
show()



# завантаження зображення, формування картинки відтінків сірого, зменьшення різкості
image = cv2.imread("L10_example.jpg")
plt.imshow(image)
plt.show()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()
gray = cv2.GaussianBlur(gray, (3, 3), 0)
plt.imshow(gray)
plt.show()
cv2.imwrite("L10_1_gray.jpg", gray)
# розпізнавання контурів
edged = cv2.Canny(gray, 10, 250)
cv2.imwrite("L10_2_edged.jpg", edged)
plt.imshow(edged)
plt.show()
# створення файлу проміжного результату
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("L10_3_closed.jpg", closed)
plt.imshow(closed)
plt.show()
# пошук контурів на зображені з підрахунком кількості книжок
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
total = 0


cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
total = 0
# ГОЛОВНА ідея РОЗПІЗНАВАННЯ ОБРАЗІВ
# цикл по контурам
for c in cnts:
    # апроксимація (згладжування) контуру
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # головна задача ідентифікації
    # якщо у контура 4 вершини то це книга
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        total += 1

# результуюче зображення
print("Знайдено {0} книги на цій картинці".format(total))
cv2.imwrite("L10_4_output.jpg", image)

plt.imshow(image)
plt.show()


