import cv2
import numpy as np


def translateImg():

    image = cv2.imread('image.jpg')

    h, w = image.shape[:2]

    shiftW, shiftH = w // 4, 10

    # Translation matrix
    translation_matrix = np.float32([[1, 0, shiftW], [0, 1, shiftH]])

    # Apply translation to the input image using OpenCV
    translated_image = cv2.warpAffine(
        src=image, M=translation_matrix, dsize=(w, h))

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/translateImg.jpg', translated_image)


def rotateImg():
    image = cv2.imread('image.jpg')

    h, w = image.shape[:2]

    # Define the rotation angle in degrees
    angle = 45

    # Define the scale factor for the rotation
    scale = 1

    # Calculate the rotation matrix using the cv2.getRotationMatrix2D function
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

    # Apply the rotation to the image using cv2.warpAffine
    rotated = cv2.warpAffine(image, matrix, (w, h))

    # Display the rotated image
    # cv2.imshow('Rotated Image', rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/rotateImg.jpg', rotated)


def scaleImg():

# Load the input image
    image = cv2.imread('image.jpg')

    h, w = image.shape[:2]

    # Calculate the nearest power of 2 that is greater than or equal to the maximum dimension of the input image
    max_dim = max(h, w)
    new_dim = 2 ** (int.bit_length(max_dim) - 1)
    if new_dim < max_dim:
        new_dim *= 2

    # Resize the input image to the new dimensions using OpenCV
    resized_img = cv2.resize(image, (new_dim, new_dim))

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/scaledImg.jpg', resized_img)


scaleImg()

rotateImg()

translateImg()


def getRGB():

    img = cv2.imread('image.jpg')

    b, g, r = cv2.split(img)

    print(r, g, b)
getRGB()    


def drawImg():

    # read image
    img = cv2.imread('image.jpg')

    # Draw a circle on the image
    new_img = cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 500, (0, 0, 255), thickness=50)

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/circledImg.jpg', new_img)

drawImg()

def bwImg():
    image = cv2.imread('image.jpg')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/bwImg.jpg', gray_image)

bwImg()

def flipImg():
    image = cv2.imread('image.jpg')

    # Flip the image horizontally using cv2.flip()
    flipped_img = cv2.flip(image, 0)

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/flippedImg.jpg', flipped_img)

flipImg()


def cropImg():
    image = cv2.imread('image.jpg')

    # Define the coordinates of the ROI
    x, y, width, height = 100, 100, 2000, 2000

    # Crop the image using numpy slicing
    cropped_image = image[y:y+height, x:x+width]
    
    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/croppedImg.jpg', cropped_image)
cropImg()

def invertImg():
    image = cv2.imread('image.jpg')

    # Invert the colors
    inverted_image = 255 - image

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/invertedImg.jpg', inverted_image)

invertImg()


def blurimg():
# Load the image
    image = cv2.imread("image.jpg")

# Apply Gaussian blur filter with kernel size of 15x15
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)

# Display the blurred image
# cv2.imshow("Blurred Image", blurred_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the blurred image
    cv2.imwrite("./Pgm2output/blurimage.jpg", blurred_img)
blurimg()   


def redimg():
    # Load the image
    image = cv2.imread("image.jpg")

# Create a red color mask
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = 255

# Multiply the image with the red color mask
    red_img = cv2.bitwise_and(image, red_mask)

# Save the red image
    cv2.imwrite("./Pgm2output/redimage.jpg", red_img)

redimg()


def greenimg():
    # Load the image
    image = cv2.imread("image.jpg")

# Create a green color mask
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = 255

# Multiply the image with the green mask
    green_img = cv2.bitwise_and(image, green_mask)

# Save the red image
    cv2.imwrite("./Pgm2output/greenimage.jpg", green_img)

greenimg() 


def blueimg():
    # Load the image
    image = cv2.imread("image.jpg")

# Create a blue color mask
    blue_mask = np.zeros_like(image)
    blue_mask[:, :, 0] = 255

# Multiply the image with the blue color mask
    blue_img = cv2.bitwise_and(image, blue_mask)

# Save the red image
    cv2.imwrite("./Pgm2output/blueimage.jpg", blue_img)

blueimg()


def rgbtohsv():
# Load the image in RGB color space
    image = cv2.imread("image.jpg")

# Convert the image to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Display the HSV image
# cv2.imshow("HSV Image", hsv_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the HSV image
    cv2.imwrite("./Pgm2output/RGBtoHsv.jpg", hsv_img)
rgbtohsv()


def rgbtolab():
# Load the image in RGB color space
    image = cv2.imread("image.jpg")

# Convert the image to Lab color space
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# Display the HSV image
# cv2.imshow("HSV Image", hsv_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the HSV image
    cv2.imwrite("./Pgm2output/RGBtoLab.jpg", lab_img)
rgbtolab()


def median_blur():
# Load the image
    image = cv2.imread("image.jpg")

# Apply median blur to the image
    median_img = cv2.medianBlur(image, 5) # change 5 to adjust the kernel size

# Display the original and median blur images
# cv2.imshow("Original Image", img)
# cv2.imshow("Median Blur Image", median_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the median blur image
    cv2.imwrite("./Pgm2output/medianblur_image.jpg", median_img)

median_blur()    





