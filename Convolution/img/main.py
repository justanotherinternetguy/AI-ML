import numpy as np
import cv2

# https://en.wikipedia.org/wiki/Kernel_(image_processing)

# SHARPEN
# KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# EDGE DET
# KERNEL = np.array([[-1, -1, -1], [-1, 4, -1],[-1, -1, -1]])
# KERNEL = np.array([[-1, -1, -1], [-1, 8, -1],[-1, -1, -1]])

# BOX BLUR
# KERNEL = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0

# GAUSSIAN BLUR 3x3
KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0

# IDENTITY
# KERNEL = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


# grayscale
def process_image(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    print("image matrix size: {0}".format(image.shape))
    print("first 5 cols and 5 rows of image mat: ", image[:5, :5])
    cv2.imwrite("topleft.jpg", image[:100, :100])
    return image

# 2d convolve
def convolve_2D(image, kernel):
    # flip kernel
    kernel = np.flipud(np.fliplr(kernel))

    # convolve output
    output = np.zeros_like(image)

    # zero padding to avoid index error
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image


    # iterate per pixel
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # process kernel w/ image
            output[y, x]=(kernel * image_padded[y: y+3, x: x+3]).sum()

    return output


input_image = process_image('cat.jpg')
out = convolve_2D(input_image, kernel=KERNEL)
cv2.imwrite('processed.jpg', out)
