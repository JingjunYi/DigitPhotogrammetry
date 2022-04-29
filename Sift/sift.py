import os
import cv2
from functools import cmp_to_key
import numpy as np


class Sift():
    
    ### Main Function ###
    def __init__(self, sigma, interval_num, blur, border_width):
        self.float_tolerance = 1e-7
        self.sigma = sigma
        self.interval_num = interval_num
        self.blur = blur
        self.border_width = border_width
        
    def detect_keypoints(self, img):
        img = img.astype('float32')
        self.base = self.gen_base(img)
        self.octave_num = self.cal_octave()
        self.gauss_kernels = self.gen_gausskernel()
        self.gauss_imgs = self.gen_gaussimg(self.base)
        self.dog_imgs = self.gen_dogimg()
        self.keypoints = self.loc_extrema()
        self.keypoints = self.del_dup()
        self.keypoints = self.trans_insize()
        
    def compute_desc(self):
        self.desc = self.gen_desc()
       
       
    ### Construct Pyramid ###
    # Generate image base of pyramid from input image by upsampling by 2 and gaussian blurring
    def gen_base(self, img):
        img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sigma_d = np.sqrt(max((self.sigma ** 2) - ((2 * self.blur) ** 2), 0.01))
        base = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_d, sigmaY=sigma_d) 
        return base 
    
    # Compute number of octaves in image pyramid
    def cal_octave(self):
        octave_num = int(round(np.log(min((self.base).shape)) / np.log(2) - 1))
        return octave_num
    
    # Generate list of gaussian kernels for constructing image pyramid.
    def gen_gausskernel(self):
        octave_imgnum = self.interval_num + 3
        k = 2 ** (1. / self.interval_num)
        # scale of gaussian blur necessary to go from one blur scale to the next within an octave
        gauss_kernels = np.zeros(octave_imgnum)
        gauss_kernels[0] = self.sigma

        for i in range(1, octave_imgnum):
            # calculate previous sigma
            sigma_p = (k ** (i - 1)) * self.sigma
            # calculate total sigma
            sigma_t = k * sigma_p
            gauss_kernels[i] = np.sqrt(sigma_t ** 2 - sigma_p ** 2)
        return gauss_kernels
    
    # Generate gaussian image pyramid
    def gen_gaussimg(self, img):
        gauss_imgs = []
        for i in range(self.octave_num):
            octave_gauss_imgs = []
            # first image in octave no need blurring
            octave_gauss_imgs.append(img)
            for gauss_kernel in self.gauss_kernels[1:]:
                img = cv2.GaussianBlur(img, (0, 0), sigmaX=gauss_kernel, sigmaY=gauss_kernel)
                octave_gauss_imgs.append(img)
            gauss_imgs.append(octave_gauss_imgs)
            octave_base = octave_gauss_imgs[-3]
            img = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)
        return np.array(gauss_imgs, dtype=object)
    
    # Generate difference-of-gaussians image pyramid
    def gen_dogimg(self):
        dog_imgs = []
        for octave_gauss_imgs in self.gauss_imgs:
            octave_dog_imgs = []
            for img1, img2 in zip(octave_gauss_imgs, octave_gauss_imgs[1:]):
                octave_dog_imgs.append(np.subtract(img2, img1))  # ordinary subtraction will not work because the images are unsigned integers
            dog_imgs.append(octave_dog_imgs)
        return np.array(dog_imgs, dtype=object)
    
    
    ### Locate Extrema ###
    # Locate coordinates of scale space extrema in the image pyramid
    def loc_extrema(self, threshold_c=0.04):
        threshold = np.floor(0.5 * threshold_c / self.interval_num * 255)
        keypoints = []
        for octave_ind, octave_dog_imgs in enumerate(self.dog_imgs):
            for img_ind, (img1, img2, img3) in enumerate(zip(octave_dog_imgs, octave_dog_imgs[1:], octave_dog_imgs[2:])):
                for i in range(self.border_width, img1.shape[0] - self.border_width):
                    for j in range(self.border_width, img1.shape[1] - self.border_width):
                        if self.judge_extrema(img1[i-1:i+2, j-1:j+2], img2[i-1:i+2, j-1:j+2], img3[i-1:i+2, j-1:j+2], threshold):
                            loc_result = self.loc_fit(i, j, img_ind + 1, octave_ind, octave_dog_imgs, threshold_c)
                            if loc_result is not None:
                                keypoint, loc_img_ind = loc_result
                                keypoints_orientations = self.cal_orientations(keypoint, octave_ind, self.gauss_imgs[octave_ind][loc_img_ind])
                                for keypoint_with_orientation in keypoints_orientations:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints
    
    # Judge if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors
    def judge_extrema(self, img1, img2, img3, threshold):
        center_pixel_value = img2[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= img1) and \
                       np.all(center_pixel_value >= img3) and \
                       np.all(center_pixel_value >= img2[0, :]) and \
                       np.all(center_pixel_value >= img2[2, :]) and \
                       center_pixel_value >= img2[1, 0] and \
                       center_pixel_value >= img2[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= img1) and \
                       np.all(center_pixel_value <= img3) and \
                       np.all(center_pixel_value <= img2[0, :]) and \
                       np.all(center_pixel_value <= img2[2, :]) and \
                       center_pixel_value <= img2[1, 0] and \
                       center_pixel_value <= img2[1, 2]
        return False
    
    # Refine coordinates of extrema based on quadratic fit
    def loc_fit(self, i, j, img_ind, octave_ind, octave_dog_imgs, threshold_c, eigen_ratio=10, max_iter=5):
        extrema_outside = False
        img_shape = octave_dog_imgs[0].shape
        for iter_ind in range(max_iter):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            img1, img2, img3 = octave_dog_imgs[img_ind-1:img_ind+2]
            pixel_cube = np.stack([img1[i-1:i+2, j-1:j+2],
                                img2[i-1:i+2, j-1:j+2],
                                img3[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.cal_grad(pixel_cube)
            hessian = self.cal_hessian(pixel_cube)
            extrema_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extrema_update[0]) < 0.5 and abs(extrema_update[1]) < 0.5 and abs(extrema_update[2]) < 0.5:
                break
            j += int(round(extrema_update[0]))
            i += int(round(extrema_update[1]))
            img_ind += int(round(extrema_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < self.border_width or i >= img_shape[0] - self.border_width or j < self.border_width or j >= img_shape[1] - self.border_width or img_ind < 1 or img_ind > self.interval_num:
                extrema_outside = True
                break
        if extrema_outside:
            # print('Updated extrema moved outside of image before reaching convergence. Skipping...')
            return None
        if iter_ind >= max_iter - 1:
            # print('Exceeded maxima number of attempts without reaching convergence for this extrema. Skipping...')
            return None
        pred_extrema = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extrema_update)
        if abs(pred_extrema) * self.interval_num >= threshold_c:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and eigen_ratio * (xy_hessian_trace ** 2) < ((eigen_ratio + 1) ** 2) * xy_hessian_det:
                # construct opencv KeyPoint object
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((j + extrema_update[0]) * (2 ** octave_ind), (i + extrema_update[1]) * (2 ** octave_ind))
                keypoint.octave = octave_ind + img_ind * (2 ** 8) + int(round((extrema_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = self.sigma * (2 ** ((img_ind + extrema_update[2]) / np.float32(self.interval_num))) * (2 ** (octave_ind + 1)) 
                keypoint.response = abs(pred_extrema)
                return keypoint, img_ind
        return None
    
    # Approximate gradient at center pixel [1, 1, 1] of 3x3x3 pixel cube using central difference formula
    def cal_grad(self, pixel_cube):
        dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
        dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
        ds = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
        return np.array([dx, dy, ds])

    # Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 pixel cube using central difference formula
    def cal_hessian(self, pixel_cube):
        center_pixel_value = pixel_cube[1, 1, 1]
        dxx = pixel_cube[1, 1, 2] - 2 * center_pixel_value + pixel_cube[1, 1, 0]
        dyy = pixel_cube[1, 2, 1] - 2 * center_pixel_value + pixel_cube[1, 0, 1]
        dss = pixel_cube[2, 1, 1] - 2 * center_pixel_value + pixel_cube[0, 1, 1]
        dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
        dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
        dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                         [dxy, dyy, dys], 
                         [dxs, dys, dss]])
        

    ### Keypoint Orientations ###
    # Calculate orientations for each keypoint
    def cal_orientations(self, keypoint, octave_ind, gauss_img, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        keypoints_orientations = []
        img_shape = gauss_img.shape
        # compare with keypoint.size obtained in loc_fit
        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_ind + 1))
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_ind))) + i
            if region_y > 0 and region_y < img_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_ind))) + j
                    if region_x > 0 and region_x < img_shape[1] - 1:
                        dx = gauss_img[region_y, region_x + 1] - gauss_img[region_y, region_x - 1]
                        dy = gauss_img[region_y - 1, region_x] - gauss_img[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2)) 
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                # quadratic peak interpolation
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < self.float_tolerance:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_orientations.append(new_keypoint)
        return keypoints_orientations
    
    
    ### Delete Duplicate ###
    # Sort keypoints and delete duplicate keypoints
    def del_dup(self):
        if len(self.keypoints) < 2:
            return self.keypoints

        self.keypoints.sort(key=cmp_to_key(self.compare_keypoints))
        unique_keypoints = [self.keypoints[0]]

        for next_keypoint in self.keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
               last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
               last_unique_keypoint.size != next_keypoint.size or \
               last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints
    
    # judge if keypoint1 is less than keypoint2 to sort
    def compare_keypoints(self, keypoint1, keypoint2):
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id
    
    
    ### Scale Transformation ###
    # Transform keypoint point, size, and octave to input image size
    def trans_insize(self):
        trans_keypoints = []
        for keypoint in self.keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            trans_keypoints.append(keypoint)
        return trans_keypoints
    
    
    ### Generate Descriptor ###
    # Generate descriptors for each keypoint
    def gen_desc(self, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        descriptors = []

        for keypoint in self.keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            gauss_img = self.gauss_imgs[octave + 1, layer]
            num_rows, num_cols = gauss_img.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            # first two dimensions are increased by 2 to account for border effects
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            # descriptor window size (described by half_width)
            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            # sqrt(2) corresponds to diagonal length of a pixel
            half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            # ensure half_width lies within image
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gauss_img[window_row, window_col + 1] - gauss_img[window_row, window_col - 1]
                            dy = gauss_img[window_row - 1, window_col] - gauss_img[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight =np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                # smoothing via trilinear interpolation
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            # remove histogram borders
            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            # threshold and normalize descriptor_vector
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_tolerance)
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')
    
    # Compute octave, layer, and scale from a keypoint
    def unpack_octave(self, keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale