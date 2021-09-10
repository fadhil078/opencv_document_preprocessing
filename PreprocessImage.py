import cv2
import os
import numpy as np
import random
from tkinter import *
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile
root = Tk()
root.title("Image Preprocessing")

# Initialization
result_norm_planes = []
arrayPoints = np.zeros((4, 2), int)
counter = 0

# Fungsi untuk event click
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global counter
        #cv2.circle(image, (x,y), 5, (255,0,0), -1)
        arrayPoints[counter] = x, y
        counter = counter + 1
        print(x, y)

def denoise_salt_and_pepper():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imwrite('Resources/01_denoise_s&p.png', dst)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()

def denoise_gaussian():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoisingColored(img,None,15,15,7,21)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(sharpen)
    plt.show()

def start_preprocessing_skewed():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    global click
    global counter
    global arrayPoints
    global result_norm_planes
    print(root.file)
    img = cv2.imread(root.file)
    width = 600
    height = 800
    dim = (width, height)
    cv2.imshow('test', img)
    cv2.waitKey(0)

    # mengkonversi gambar menjadi grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menginvert grayscale image menggunakan bitwise_not
    gray_img = cv2.bitwise_not(gray_img)

    # Memilih koordinat x dan y dari nilai pixel yg lebih besar dari 0
    # dengan menggunakan method column_stack dari numpy
    coordinates = np.column_stack(np.where(gray_img > 0))

    # Menghitung sudut kemiringan skew document menggunakan method minAreaRect()
    # dari cv2 yang akan mengembalikan nilai sudut dengan range -90 sampai 0 derajat
    # (dimana 0 tidak termasuk)

    ang = cv2.minAreaRect(coordinates)[-1]

    # Sudut yang terotasi dari text region akan di store pada variable sudut
    # dibawah ini adalah kondisi untuk angle, jika text region angle kurang dari -45, maka tambahkan 90 derajat
    # jika tidak, maka kita mengalikan sudut dengan minus untuk membuat sudut menjadi positif

    if ang < -45:
        ang = -(90 + ang)
    else:
        ang = -ang
    print(ang)

    # Menghitung center dari text region
    height, width = img.shape[:2]
    center_img = (width / 2, height / 2)

    # Setelah mendapatkan sudut dari kemiringan text, terapkan getRotationMatrix2D()
    # untuk mendapatkan rotation matrix gunakan method wrapAffine()
    # untuk rotate sudut

    rotationMatrix = cv2.getRotationMatrix2D(center_img, ang, 1.0)
    rotated_img = cv2.warpAffine(img, rotationMatrix, (width+30, height), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite('Resources/rotated.jpg', rotated_img)


def start_preprocessing_photos():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    global click
    global counter
    global arrayPoints
    global result_norm_planes

    print(root.file)
    image = cv2.imread(root.file)
    image = cv2.resize(image, (600, 800))
    # Melakukan pointing posisi dengan event mouse click
    while True:
        if counter == 4:
            point1 = arrayPoints[1, 0] - arrayPoints[0, 0]
            point2 = arrayPoints[3, 1] - arrayPoints[0, 1]
            pts1 = np.float32([arrayPoints[0], arrayPoints[1], arrayPoints[2], arrayPoints[3]])
            pts2 = np.float32([[0, 0], [point1, 0], [0, point2], [point1, point2]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            final = cv2.warpPerspective(image, matrix, (point1, point2))
            rgb_planes = cv2.split(final)

            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8UC1)
                result_norm_planes.append(norm_img)

            result_norm = cv2.merge(result_norm_planes)
            cv2.imwrite('Resources/result_norm.jpg', result_norm)
            break

        cv2.namedWindow('Input')
        cv2.setMouseCallback('Input', click)
        cv2.imshow('Input', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

def salt_and_pepper():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row,col,ch = img.shape

    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(img)

    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int (num_salt))
            for i in img.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img.shape]
    out[coords] = 0
    
    cv2.imwrite('Resources/01_s&p.png',out)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(out),plt.title('S&P Noise')
    plt.xticks([]), plt.yticks([])
    plt.show()

def gaussian_noise():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img,gauss)

    cv2.imwrite('Resources/01_gn.png',img_gauss)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_gauss),plt.title('Gaussian Noise')
    plt.xticks([]), plt.yticks([])
    plt.show()

def Poisson():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gauss = np.random.poisson(1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')     
    noisy = img + (img * gauss)
    cv2.imwrite('Resources/01_p.png',noisy)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(noisy),plt.title('Poisson Noise')
    plt.xticks([]), plt.yticks([])
    plt.show()

def speckle():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')     
    noisy = img + (img * gauss)
    cv2.imwrite('Resources/01_s.png',noisy)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(noisy),plt.title('Speckle Noise')
    plt.xticks([]), plt.yticks([])
    plt.show()

#Menubar
menu = Menu(root)
root.config(menu=menu)

#Submenu File
preprocessmenu = Menu(menu, tearoff=0)
addnoise = Menu(menu, tearoff=0)
menu.add_cascade(label="Preprocess", menu=preprocessmenu)
menu.add_cascade(label="Add Noise", menu=addnoise)
menu.add_cascade(label="Exit", command=root.quit)

denoise=Menu(preprocessmenu, tearoff=0)
denoise.add_cascade(label="Salt and Pepper", command=denoise_salt_and_pepper)
denoise.add_cascade(label="gaussian", command=denoise_gaussian)

preprocessmenu.add_cascade(label="Skewed", command=start_preprocessing_skewed)
preprocessmenu.add_cascade(label="Photos", command=start_preprocessing_photos)
preprocessmenu.add_cascade(label="Denoise", menu=denoise)

addnoise.add_cascade(label="Salt and Pepper", command=salt_and_pepper)
addnoise.add_cascade(label="Gaussian Noise", command=gaussian_noise)
addnoise.add_cascade(label="Poisson", command=Poisson)
addnoise.add_cascade(label="Speckle", command=speckle)


root.mainloop()