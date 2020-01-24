import numpy as np
import random
import cv2

# this script contains data augmentation code
# use moderate_combo or heavy_combo
# RotNetDataGenerator inputs moderate_combo or heavy_combo

def uniform_noise(image):
    h,w,c = image.shape
    total = w * h * 3
    noise = np.random.randint(-30, 31, size=total)
    noise = noise.reshape(h, w, 3)
    image = image+noise
    image = np.clip(image,0,255)

    return image

def gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 3
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    image = image+gauss
    image = np.clip(image,0,255)

    return image


def add_contrast_brightness_heavy(image):

    contrast = np.random.uniform(0.9, 1.1)
    brightness = np.random.randint(-5, 5)
    image = image * contrast + brightness
    image.astype(np.uint8)
    image = np.clip(image, 0, 255)

    return image

def gaussian_blur(image):
    image = np.clip(image, 0, 255)
    std = np.random.uniform(1, 3.5)
    image = cv2.GaussianBlur(image, (5, 5), std)
    image = np.clip(image, 0, 255)

    return image

def moderate_combo(image):
    image = gaussian_blur(image)
    image = add_contrast_brightness_heavy(image)
    image = uniform_noise(image)
    image = gaussian_noise(image)
    return image

def heavy_combo(image):
    def add_noise_heavy(image):

        image = np.clip(image, 0, 255)
        IMAGE_HEIGHT,IMAGE_WIDTH,C = image.shape
        if bool(random.getrandbits(1)):  # UNIFORM NOISE
            total = IMAGE_WIDTH * IMAGE_HEIGHT * 3
            a = np.random.randint(-30, 31, size=total)
            # Includes low but not high
            a = a.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            noise = image + a
            to_int_noise = noise
            # to_int_noise = noise.astype('uint8')

        else:  # GAUSSIAN NOISE
            row, col, ch = image.shape
            mean = 0
            sigma = 3
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            # code for plotting
            # count, bins, ignored = plt.hist(gauss, 30, density=True)
            # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
            #          np.exp(- (bins - mean) ** 2 / (2 * sigma ** 2)),
            #          linewidth=2, color='r')
            # plt.show()
            # code for plotting

            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            to_int_noise = noisy
            # to_int_noise = noisy.astype('uint8')

        to_int_noise = np.clip(to_int_noise, 0, 255)

        return to_int_noise

    def add_contrast_brightness_heavy(image):
        image = np.clip(image, 0, 255)
        contrast = np.random.uniform(0.9, 1.1)
        brightness = np.random.randint(-5, 5)
        adjusted_image = image * contrast + brightness
        to_int_adjusted = adjusted_image
        # to_int_adjusted = adjusted_image.astype('uint8')

        to_int_adjusted = np.clip(to_int_adjusted, 0, 255)
        return to_int_adjusted

    def blur_randomly_heavy(image):
        image = np.clip(image, 0, 255)
        std = np.random.uniform(1, 3.5)
        blurred = cv2.GaussianBlur(image, (5, 5), std)
        blurred = np.clip(blurred, 0, 255)

        return blurred

    image = blur_randomly_heavy(image)
    image = add_noise_heavy(image)
    image = add_contrast_brightness_heavy(image)
    return image