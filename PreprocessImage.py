from tkinter import *
import subprocess
import cv2
import os
import re
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
from tkinter.filedialog import askopenfilename, askopenfilenames
from tkinter.filedialog import asksaveasfile
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\Tesseract.exe"
root = Tk()
root.title("Image Preprocessing")
root.geometry('400x400')

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

#=========================================================================Fungsi Filtering==============================================================================#

def gaussian_filtering():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img,(5,5),0)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Gaussian Filtering')
    plt.xticks([]), plt.yticks([])
    plt.show()

def averaging_filtering():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.blur(img,(5,5))
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Mean Filtering')
    plt.xticks([]), plt.yticks([])
    plt.show()

def median_filtering():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.medianBlur(img,5)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Median Filtering')
    plt.xticks([]), plt.yticks([])
    plt.show()

def bilateral_filtering():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.bilateralFilter(img,9,75,75)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Bilateral Filtering')
    plt.xticks([]), plt.yticks([])
    plt.show()

#==========================================================================Fungsi Denoise===============================================================================#
def preprocessing_denoise(image):
    image_gray = image.copy()

    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    rows, columns,_ = image.shape
    range_rows = rows - 1
    range_columns = columns - 1

    print(image_gray[200,200])
    thresh = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    for index in range(0, range_rows):
        for index_2 in range(0, range_columns):
            if thresh[index, index_2] < 255:
                if thresh[index + 1, index_2] > 0 and thresh[index - 1, index_2] > 0 and thresh[index, index_2 - 1] > 0 and thresh[index, index_2 + 1] > 0:
                    image_gray[index, index_2] = 255

    thresh_2 = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    for index in range(0, range_rows):
        for index_2 in range(0, range_columns):
            if thresh_2[index, index_2] < 255:
                if thresh_2[index + 1, index_2] > 0 and thresh_2[index - 1, index_2] > 0:
                    image_gray[index, index_2] = 255
                elif thresh_2[index, index_2 - 1] > 0 and thresh[index, index_2 + 1] > 0:
                    image_gray[index, index_2] = 255

    return image_gray

def denoise_salt_and_pepper(image):
    output_image = cv2.fastNlMeansDenoisingColored(image,None,15,10,7,21)
    #gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    #thresh, im_bw = cv2.threshold(gray, 200, 1000, cv2.THRESH_BINARY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output_image = cv2.filter2D(output_image, -1, sharpen_kernel)
    return output_image

def denoise_gaussian(image):
    dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    blur = cv2.medianBlur(dst,5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output_image = cv2.filter2D(blur, -1, sharpen_kernel)
    return output_image


#========================================================================Fungsi untuk deskew document===================================================================#
def warp_affine(angle, image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    print("height = " + str(h))
    print("width = " + str(w))
    print("center = " + str(center))
    print("M = ")
    print(M)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))
    return rotated

def rotate_skewed_right(angle, image):
    if angle < -45:
        angle = -(90 + angle)
        angle = 180 + (angle - 90)
    else:
        angle = -(angle)
        angle = 180 + (angle - 90)
    # rotate the image to deskew it
    print("new angle = ")
    print(angle)
    rotated_image = warp_affine(angle,image)
    return rotated_image

def rotate_skewed_left(angle, image):
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -(angle)
    # rotate the image to deskew it
    rotated_image = warp_affine(angle,image)
    return rotated_image

def rotate_skewed_right_upside_down(angle, image):
    if angle < 0:
        angle = -(90 + angle)
        angle = 360 + (angle - 90)
    else:
        angle = -(angle)
        angle = 360 + (angle - 90)

    rotated_image = warp_affine(angle,image)
    return rotated_image

def rotate_skewed_left_upside_down(angle, image):
    if angle < 0:
        angle = -(90 + angle)
        angle = angle + 180
    else:
        angle = -(angle)
        angle = angle + 180
    # rotate the image to deskew it
    rotated_image = warp_affine(angle,image)
    return rotated_image
    
def rotate_skewed_horizontal_right(image):
    img_rotate_90_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_rotate_90_counterclockwise

def rotate_skewed_horizontal_left(image):
    img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return img_rotate_90_clockwise

def rotate_upside_down(image):
    img_rotate_180 = cv2.rotate(image, cv2.ROTATE_180)
    return img_rotate_180

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def preprocess_skewed(image, denoised):
    global click
    global counter
    global arrayPoints
    global result_norm_planes
    #cv2.imshow("Gray", img)
    if denoised == 0:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold to make image black and white
    ret, img_thresh = cv2.threshold(image, 200,255, cv2.THRESH_BINARY)
    #cv2.imshow("Black White", img)
    coords = np.column_stack(np.where(img_thresh < 255))
    angle = cv2.minAreaRect(coords)[-1]
    # Print angle
    print('angle = ' + str(angle))

    black_coords = np.column_stack(np.where(img_thresh == 0))

    # Koordinat untuk nilai height terendah (letak teratas)
    smallest_coords_height = np.amin(black_coords[:,0])

    # Koordinat untuk nilai height tertinggi (letak terbawah)
    largest_coords_height = np.amax(black_coords[:,0])

    # Koordinat untuk nilai width terendah (letak teratas)
    smallest_coords_width = np.amin(black_coords[:,1])

    # Koordinat untuk nilai width tertinggi (letak terbawah)
    largest_coords_width = np.amax(black_coords[:,1])

    # Mencari koordinat width pada nilai koordinat height terendah
    width_smallest_height = []
    for index, val in np.ndenumerate(black_coords[:,0]):
        if val == smallest_coords_height:
            #print("\nNilai height terendah = ")
            #print(black_coords[index,0])

            #print("\nNilai width pada height terendah = ")
            #print(black_coords[index,1])
            width_smallest_height.append(black_coords[index,1])

    wsh = np.concatenate(width_smallest_height)
    wsh_largest = max(wsh)
    print("\nkoordinat x dan y pada smallest coords height = [" + str(smallest_coords_height) + ", " + str(wsh_largest) + "]")

    # Mencari koordinat width pada nilai koordinat height tertinggi
    width_largest_height = []
    for index, val in np.ndenumerate(black_coords[:,0]):
        if val == largest_coords_height:
            #print("\nNilai height tertinggi = ")
            #print(black_coords[index,0])

            #print("\nNilai width pada height tertinggi = ")
            #print(black_coords[index,1])
            width_largest_height.append(black_coords[index,1])

    wlh = np.concatenate(width_largest_height)
    wlh_largest = max(wlh)
    print("\nkoordinat x dan y pada largest coords height = [" + str(largest_coords_height) + ", " + str(wlh_largest) + "]")

    # Mencari koordinat height pada nilai koordinat width terendah
    height_smallest_width = []
    for index, val in np.ndenumerate(black_coords[:,1]):
        if val == smallest_coords_width:
            #print("\nNilai width terendah = ")
            #print(black_coords[index,1])

            #print("\nNilai height pada width terendah = ")
            #print(black_coords[index,0])
            height_smallest_width.append(black_coords[index,0])

    hsw = np.concatenate(height_smallest_width)
    hsw_largest = max(hsw)
    print("\nkoordinat x dan y pada smallest coords width = [" + str(hsw_largest) + ", " + str(smallest_coords_width) + "]")

    # Mencari koordinat height pada nilai koordinat width tertinggi
    height_largest_width = []
    for index, val in np.ndenumerate(black_coords[:,1]):
        if val == largest_coords_width:
            # print("\nNilai width tertinggi = ")
            # print(black_coords[index,1])

            # print("\nNilai height pada width tertinggi = ")
            # print(black_coords[index,0])
            height_largest_width.append(black_coords[index,0])

    hlw = np.concatenate(width_largest_height)
    hlw_largest = max(hlw)
    print("\nkoordinat x dan y pada largest coords width = [" + str(hlw_largest) + ", " + str(largest_coords_width) + "]")

    width = largest_coords_width - smallest_coords_width
    height = largest_coords_height - smallest_coords_height
    print(width)
    print(height)
    print("Angle : " + str(angle))
    if 1 < angle < 88:
        if hsw_largest > hlw_largest:
            image = rotate_skewed_right(angle, image)
        elif hsw_largest < hlw_largest:
            image = rotate_skewed_left(angle, image)
    else:
        if width > height:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    '''thresh_rot = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('thresh rot', thresh_rot)
    cv2.waitKey(0)
    white_coords_rot = np.column_stack(np.where(thresh_rot < 50))
    smallest_coords_height_rot = np.amin(white_coords_rot[:,0])
    smallest_coords_height_str_rot = str(smallest_coords_height_rot)
    print("\nSmallest white coordinate height = " + smallest_coords_height_str_rot)

    largest_coords_height_rot = np.amax(white_coords_rot[:,0])
    largest_coords_height_str_rot = str(largest_coords_height_rot)
    print("\nLargest white coordinates height = " + largest_coords_height_str_rot)

    smallest_coords_width_rot = np.amin(white_coords_rot[:,1])
    smallest_coords_width_str_rot = str(smallest_coords_width_rot)
    print("\nSmallest white coordinates width = " + smallest_coords_width_str_rot)

    largest_coords_width_rot = np.amax(white_coords_rot[:,1])
    largest_coords_width_str_rot = str(largest_coords_width_rot)
    print("\nLargest white coordinates width = " + largest_coords_width_str_rot)

    # Cut image image  [height, width]
    if smallest_coords_height_rot <= 1:
        cropped_image = image[(smallest_coords_height_rot):(largest_coords_height_rot + 5), (smallest_coords_width_rot - 5):(largest_coords_width_rot + 5)]
        cv2.imshow("cropped image", cropped_image)
    else:
        cropped_image = image[(smallest_coords_height_rot - 5):(largest_coords_height_rot + 5), (smallest_coords_width_rot - 5):(largest_coords_width_rot + 5)]
        cv2.imshow("cropped image", cropped_image)
    cv2.waitKey(0)'''
    return image

def check_orientation_layout(image, osd_rotate_angle):
    if osd_rotate_angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif osd_rotate_angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif osd_rotate_angle == 180:
        image = rotate_upside_down(image)
    return image
    print('ini adalah fungsi pengecekan apakah dokumen tersebut portrait atau landscape')

#======================================================================= OCR Test =======================================================================================$
def ocr_test():
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\Tesseract.exe"
    # open and ocr
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png")))
    img = cv2.imread(root.file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 11)
    text = pytesseract.image_to_string(adaptive_threshold)

    # save to the file and open it
    name, extension = os.path.splitext(root.file)
    file_name = os.path.split(name)[1]
    string = root.file

    osd = re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(string,config='--psm 0 -c min_characters_to_try=5')).group(0)
    print("root file = " + string)
    print("file name = " + file_name)
    print("Sudut = " + str(osd))
    folder = 'ocr_test_result/'
    save_file_name = folder + file_name + '_ocr_result.txt'

    with open(save_file_name, 'w') as f:
        f.write(text)
    
    print("Dibawah ini merupakan hasil open file text\n")
    with open(save_file_name, 'r') as f:
        isi = f.read()
        print(isi)
    


#=======================================================================Preprocessing untuk auto-cropped=================================================================#
def preprocess_autocropped(image):
    # gray_rot = cv2.bitwise_not(image)
    thresh_rot = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('thresh rot', thresh_rot)
    cv2.waitKey(0)
    white_coords_rot = np.column_stack(np.where(thresh_rot == 255))
    smallest_coords_height_rot = np.amin(white_coords_rot[:,0])
    smallest_coords_height_str_rot = str(smallest_coords_height_rot)
    print("\nSmallest white coordinate height = " + smallest_coords_height_str_rot)

    largest_coords_height_rot = np.amax(white_coords_rot[:,0])
    largest_coords_height_str_rot = str(largest_coords_height_rot)
    print("\nLargest white coordinates height = " + largest_coords_height_str_rot)

    smallest_coords_width_rot = np.amin(white_coords_rot[:,1])
    smallest_coords_width_str_rot = str(smallest_coords_width_rot)
    print("\nSmallest white coordinates width = " + smallest_coords_width_str_rot)

    largest_coords_width_rot = np.amax(white_coords_rot[:,1])
    largest_coords_width_str_rot = str(largest_coords_width_rot)
    print("\nLargest white coordinates width = " + largest_coords_width_str_rot)

    # Cut image image  [height, width]
    cropped_image = image[(smallest_coords_height_rot):(largest_coords_height_rot), (smallest_coords_width_rot):(largest_coords_width_rot)]
    return cropped_image

#=======================================================================Preprocessing untuk foto=========================================================================#
def preprocess_photos():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png")))
    global click
    global counter
    global arrayPoints
    global result_norm_planes
    name, extension = os.path.splitext(root.file)
    file_name = os.path.split(name)[1]
    print(root.file)
    image = cv2.imread(root.file)
    rows, columns, _ = image.shape
    print("Rows = ", rows)
    print("Columns = ", columns)
    image = cv2.resize(image, (columns, rows))
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

            save_file_name = 'Resources/photos_preprocessed/' + file_name + '_photopprocessed' + extension
            print("Save file path = ", save_file_name)
            cv2.imwrite(save_file_name, result_norm)
            counter = 0
            break

        cv2.namedWindow('Input')
        cv2.setMouseCallback('Input', click)
        cv2.imshow('Input', image)

        #if selesai == True:
        #    break
        if cv2.waitKey(1) & 0xFF == 27:
            break

#==========================================================================Add noise Functions========================================================================#
def salt_and_pepper():
    root.file = askopenfilenames(initialdir = "", title = "Choose Document", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png")))
    last_index = len(root.file)
    for index in range(0, last_index):
        name, extension = os.path.splitext(root.file[index])
        file_name = os.path.split(name)[1]
        img = cv2.imread(root.file[index])
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

        save_file_name = 'Resources/salt_and_pepper_multi/' + file_name + '_s&p' + extension
        print("Save file path = ", save_file_name)
        cv2.imwrite(save_file_name, out)

def gaussian_noise():
    root.file = askopenfilenames(initialdir = "", title = "Choose Document", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png")))
    last_index = len(root.file)
    for index in range(0, last_index):
        name, extension = os.path.splitext(root.file[index])
        file_name = os.path.split(name)[1]
        img = cv2.imread(root.file[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gauss = np.random.normal(0,1,img.size)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv2.add(img,gauss)
        save_file_name = 'Resources/gaussian/' + file_name + '_gauss' + extension
        print("Save file path = ", save_file_name)
        cv2.imwrite(save_file_name, img_gauss)

#============================================================================Edge Detection=============================================================================#
def canny():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges, cmap='gray'),plt.title('Canny')
    plt.xticks([]), plt.yticks([])
    plt.show()

def laplacian():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(gray, (3,3), 0)
    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_image = cv2.Laplacian(img, ksize=3, ddepth=cv2.CV_16S)
    # converting back to uint8
    filtered_image = cv2.convertScaleAbs(filtered_image)

    plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(filtered_image, cmap='gray'),plt.title('Laplacian')
    plt.xticks([]), plt.yticks([])
    plt.show()

def sobel():
    root.file = askopenfilename(initialdir = "", title = "Choose Document", filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg")))
    img = cv2.imread(root.file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(gray, (3,3), 0)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # 3 X 3 X-direction kernel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_image_y = cv2.filter2D(gray, -1, sobel_y)
    filtered_image_x = cv2.filter2D(gray, -1, sobel_x)
    plt.subplot(131),plt.imshow(img,cmap = 'gray')
    plt.title('Citra Awal'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(filtered_image_x,cmap = 'gray')
    plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(filtered_image_y,cmap = 'gray')
    plt.title('Sobel y'), plt.xticks([]), plt.yticks([])
    plt.show()
#==============================================================Count Noises Function=========================================================#
def count_noise(image):
    image_gray = image.copy()
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    rows, columns,_ = image.shape
    range_rows = rows - 1
    range_columns = columns - 1
    count = 0
    thresh = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow('thresh', thresh)
    for index in range(0, range_rows):
        for index_2 in range(0, range_columns):
            if thresh[index, index_2] < 255:
                if thresh[index + 1, index_2] > 0 and thresh[index - 1, index_2] > 0 and thresh[index, index_2 - 1] > 0 and thresh[index, index_2 + 1] > 0:
                    count = count + 1

    return count

    
#====================================================================Start Preprocessing Function=======================================================================#
def start_preprocessing():
    root.file = askopenfilenames(initialdir = "", title = "Choose Document", filetypes=(("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("JPEG Files", "*.jpeg")))
    last_index = len(root.file)
    iterate = 0
    for index in range(0, last_index):
        print("test")
        name, extension = os.path.splitext(root.file[index])
        file_name = os.path.split(name)[1]
        image = cv2.imread(root.file[index])
        #cv2.imshow('before', image)
        string = root.file[index]
        print('root file = ' + string)

        # Validasi untuk denoise
        total_noise = count_noise(image)
        print("Total Noise = " + str(total_noise))
        denoised = 0
        if(total_noise > 250):
            image = preprocessing_denoise(image)
            denoised = 1

        # Validasi kemiringan gambar dokumen
        output_image = preprocess_skewed(image, denoised)
        #cv2.imshow('after', output_image)

        # Preprocess upscaling images
        sr2 = cv2.dnn_superres.DnnSuperResImpl_create()
        path2 = "ESPCN_x4.pb"
        sr2.readModel(path2)
        sr2.setModel("espcn", 4)
        result2 = sr2.upsample(output_image)
        scale_percent = 50
        width = int(result2.shape[1] * scale_percent / 100)
        height = int(result2.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(output_image, dim, interpolation = cv2.INTER_AREA)
        result2 = resized
        #result2 = output_image

        # Validasi untuk pengecekan ulang apakah gambar tersebut upside down atau tidak
        # osd = int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(string,config='--psm 0 -c min_characters_to_try=5')).group(0))
        # print("sudut file " + file_name + " = " + str(osd))

        # Menyimpan file temp 1
        save_file_name = 'temp1/' + file_name + '_result' + extension
        print(save_file_name)
        cv2.imwrite(save_file_name, result2)

        #image_copy = output_image.copy()
        
        #output_image = check_orientation_layout(image_copy, osd)
        
        thresh_rot = cv2.threshold(result2, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv2.imshow('thresh rot', thresh_rot)
        #cv2.waitKey(0)
        white_coords_rot = np.column_stack(np.where(thresh_rot < 50))
        smallest_coords_height_rot = np.amin(white_coords_rot[:,0])
        smallest_coords_height_str_rot = str(smallest_coords_height_rot)
        print("\nSmallest white coordinate height = " + smallest_coords_height_str_rot)

        largest_coords_height_rot = np.amax(white_coords_rot[:,0])
        largest_coords_height_str_rot = str(largest_coords_height_rot)
        print("\nLargest white coordinates height = " + largest_coords_height_str_rot)

        smallest_coords_width_rot = np.amin(white_coords_rot[:,1])
        smallest_coords_width_str_rot = str(smallest_coords_width_rot)
        print("\nSmallest white coordinates width = " + smallest_coords_width_str_rot)

        largest_coords_width_rot = np.amax(white_coords_rot[:,1])
        largest_coords_width_str_rot = str(largest_coords_width_rot)
        print("\nLargest white coordinates width = " + largest_coords_width_str_rot)

        cropped_image = result2[(smallest_coords_height_rot):(largest_coords_height_rot), (smallest_coords_width_rot):(largest_coords_width_rot)]
        '''if smallest_coords_height_rot <= 1:
            cropped_image = result2[(smallest_coords_height_rot):(largest_coords_height_rot + 5), (smallest_coords_width_rot - 5):(largest_coords_width_rot + 5)]
        else:
            cropped_image = result2[(smallest_coords_height_rot - 5):(largest_coords_height_rot + 5), (smallest_coords_width_rot - 5):(largest_coords_width_rot + 5)]'''
        
        # Menyimpan file temp 2
        save_file_name = 'temp2/' + file_name + '_result' + extension
        cv2.imwrite(save_file_name, cropped_image)

        # Preprocess upside down image and orientation layout (portrait, landscape)
        angle_rotate = int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(save_file_name,config='--psm 0 -c min_characters_to_try=5')).group(0))
        print("sudut file " + file_name + " = " + str(angle_rotate))
        output_image = check_orientation_layout(cropped_image, angle_rotate)

        # Menyimpan file terakhir
        save_file_name = 'Preprocessed_Result/' + file_name + '_result' + extension
        cv2.imwrite(save_file_name, output_image)
        '''if(angle_rotate == 180):
            image_upside_down = cv2.imread(save_file_name)
            output_image = rotate_upside_down(image_upside_down)
            #cv2.imshow('test upside down', output_image)
            
            # Validasi untuk autocropped dokumen 
            #output_image = preprocess_autocropped(output_image)
            #cv2.imshow('after cropped', cropped)
            cv2.imwrite(save_file_name, output_image)
        else:
            #output_image = preprocess_autocropped(output_image)
            #cv2.imshow('after cropped', output_image)
            cv2.imwrite(save_file_name, output_image)'''
        
        iterate = iterate + 1

    if iterate > 0:
        print("\nPreprocessing Successful")

#================================================================================Tkinter Menu============================================================================#
# Menubar
menu = Menu(root)
root.config(menu=menu)

# Menu of Tkinter Program
preprocessmenu = Menu(menu, tearoff=0)
addnoise = Menu(menu, tearoff=0)
edgedetection = Menu(menu, tearoff=0)
filtering = Menu(menu, tearoff=0)
menu.add_cascade(label="Preprocess", menu=preprocessmenu)
menu.add_cascade(label="Add Noise", menu=addnoise)
menu.add_cascade(label="Edge Detection", menu=edgedetection)
menu.add_cascade(label="Filtering",menu=filtering)
menu.add_cascade(label="OCR Test", command=ocr_test)
menu.add_cascade(label="Exit", command=root.quit)

# Submenu of Preprocess denoise
denoise=Menu(preprocessmenu, tearoff=0)
denoise.add_cascade(label="Salt and Pepper", command=denoise_salt_and_pepper)
denoise.add_cascade(label="gaussian", command=denoise_gaussian)

# Submenu of Preprocessmenu
preprocessmenu.add_cascade(label="Start preprocessing image", command=start_preprocessing)
preprocessmenu.add_cascade(label="Photos", command=preprocess_photos)
preprocessmenu.add_cascade(label="Denoise", menu=denoise)

# Submenu of add noise
addnoise.add_cascade(label="Salt and Pepper", command=salt_and_pepper)
addnoise.add_cascade(label="Gaussian Noise", command=gaussian_noise)

# Submenu of edge detection
edgedetection.add_cascade(label="Canny", command=canny)
edgedetection.add_cascade(label="Laplacian", command=laplacian)
edgedetection.add_cascade(label="Sobel", command=sobel)

# Submeny of Filtering
filtering.add_cascade(label="Gaussian Filtering", command=gaussian_filtering)
filtering.add_cascade(label="Averaging Filtering", command=averaging_filtering)
filtering.add_cascade(label="Median Filtering", command=median_filtering)
filtering.add_cascade(label="Bilateral Filtering", command=bilateral_filtering)

root.mainloop()

#======================================================================================================================================================================#