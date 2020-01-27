import numpy as np
import random
import cv2


def IEEE_VR_combo(image):
    def add_contrast_brightness(image):
        image = np.clip(image, 0, 255)
        contrast = np.random.uniform(0.9, 1.1)
        brightness = np.random.randint(-5, 5)
        adjusted_image = image * contrast + brightness
        to_int_adjusted = adjusted_image
        # to_int_adjusted = adjusted_image.astype('uint8')

        to_int_adjusted = np.clip(to_int_adjusted, 0, 255)
        return to_int_adjusted

    def blur_randomly(image):

        image = np.clip(image, 0, 255)
        std = np.random.uniform(0, 1.5)
        blurred = cv2.GaussianBlur(image, (5, 5), std)
        blurred = np.clip(blurred, 0, 255)

        return blurred

    def add_noise(image):
        IMAGE_HEIGHT, IMAGE_WIDTH, C = image.shape
        image = np.clip(image, 0, 255)
        if bool(random.getrandbits(1)):  # UNIFORM NOISE
            total = IMAGE_WIDTH * IMAGE_HEIGHT * 3
            a = np.random.randint(-3, 4, size=total)
            # Includes low but not high
            a = a.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            noise = image + a
            to_int_noise = noise
            # to_int_noise = noise.astype('uint8')

        else:  # GAUSSIAN NOISE
            row, col, ch = image.shape
            mean = 0
            sigma = 1
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

    if bool(random.getrandbits(1)):
        image = blur_randomly(image)
    if bool(random.getrandbits(1)):
        image = add_noise(image)
    if bool(random.getrandbits(1)):
        image = add_contrast_brightness(image)

    return image