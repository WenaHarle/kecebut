# augmint.py
import cv2
import numpy as np
import random

class augmints:
    def __init__(self):
        self.pipeline = []

    def add(self, aug_type, p=1.0, **params):
        self.pipeline.append({"type": aug_type, "p": p, "params": params})

    def apply_augmentations(self, image, bboxes=None):
        for aug in self.pipeline:
            prob = aug.get("p", 1.0)
            if random.random() < prob:
                aug_type = aug.get("type")
                params = aug.get("params", {})
                if aug_type == "rotate":
                    image, bboxes = self.rotate(image, bboxes, params)
                elif aug_type == "hue_saturation_value":
                    image = self.hue_saturation_value(image, params)
                elif aug_type == "brightness_contrast":
                    image = self.brightness_contrast(image, params)
                elif aug_type == "blur":
                    image = self.blur(image, params)
                elif aug_type == "motion_blur":
                    image = self.motion_blur(image, params)
                elif aug_type == "optical_distortion":
                    image = self.optical_distortion(image, params)
                elif aug_type == "color_jitter":
                    image = self.color_jitter(image, params)
        return image, bboxes

    def rotate(self, image, bboxes, params):
        angle = random.uniform(-params.get("limit", 45), params.get("limit", 45))
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int(h * sin_angle + w * cos_angle)
        new_h = int(h * cos_angle + w * sin_angle)
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

        rotated_bboxes = []
        if bboxes:
            for bbox in bboxes:
                x, y, width, height = bbox

                # Define the four corners of the bounding box
                corners = np.array([
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height]
                ])

                # Rotate each corner
                rotated_corners = cv2.transform(np.array([corners]), rotation_matrix)[0]

                # Calculate new bounding box from rotated corners
                x_min, y_min = rotated_corners.min(axis=0)
                x_max, y_max = rotated_corners.max(axis=0)
                rotated_bboxes.append([
                    x_min, y_min,
                    x_max - x_min, y_max - y_min
                ])

        return rotated_image, rotated_bboxes


    def hue_saturation_value(self, image, params):
        hue_shift = random.uniform(-params.get("hue_shift_limit", 0), params.get("hue_shift_limit", 0))
        sat_shift = random.uniform(-params.get("sat_shift_limit", 0), params.get("sat_shift_limit", 0))
        val_shift = random.uniform(-params.get("val_shift_limit", 0), params.get("val_shift_limit", 0))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 0] = np.clip(hsv_image[..., 0] + hue_shift, 0, 179)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] + sat_shift, 0, 255)
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] + val_shift, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def brightness_contrast(self, image, params):
        brightness = random.uniform(-params.get("brightness_limit", 0), params.get("brightness_limit", 0))
        contrast = random.uniform(-params.get("contrast_limit", 0), params.get("contrast_limit", 0))
        brightness_factor = 1 + brightness
        contrast_factor = 1 + contrast
        return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor * 50)

    def blur(self, image, params):
        blur_limit = min(max(int(params.get("blur_limit", 3)), 1), 7)
        kernel_size = max(1, blur_limit)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def motion_blur(self, image, params):
        blur_limit = min(max(int(params.get("blur_limit", 3)), 1), 7)
        kernel_size = max(1, blur_limit)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, kernel_size // 2] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def optical_distortion(self, image, params):
        h, w = image.shape[:2]
        k = np.eye(3)  # Camera matrix with focal lengths
        k[0, 0] = 1.0  # fx
        k[1, 1] = 1.0  # fy
        k[0, 2] = w / 2.0  # cx
        k[1, 2] = h / 2.0  # cy

        # Ensure distCoeffs is in the correct shape, e.g., a 1x4 zero array for no distortion
        d = np.zeros((1, 4), dtype=np.float32)  # or adjust to match required distortion parameters

        map1, map2 = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), cv2.CV_32FC1)
        distorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        return distorted_image

    def color_jitter(self, image, params):
        brightness = params.get("brightness", 0)
        contrast = params.get("contrast", 0)
        saturation = params.get("saturation", 0)
        hue = params.get("hue", 0)
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] * (1 + saturation), 0, 255)
        hsv_image[..., 0] = np.clip(hsv_image[..., 0] + hue, 0, 179)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        image = np.clip(image * (1 + brightness), 0, 255).astype(np.uint8)
        return cv2.convertScaleAbs(image, alpha=1 + contrast, beta=0)
