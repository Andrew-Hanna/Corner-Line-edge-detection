import sys
from PyQt5.QtWidgets import QApplication,QSizePolicy, QMainWindow, QLabel, QHBoxLayout,QVBoxLayout ,QWidget, QPushButton, QFileDialog, QComboBox, QSlider, QSpinBox, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize
import cv2
import numpy as np

# Define img_frame and edges_frame in the global scope
img_frame = None
edges_frame = None
img = None  # Add a global variable to store the loaded image



######################### corner######################

def sobel_operator(img):
    # Define the Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialize the gradients
    gradient_x = np.zeros_like(img, dtype=np.float64)
    gradient_y = np.zeros_like(img, dtype=np.float64)

    # Compute the gradients
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            region = img[i-1:i+2, j-1:j+2]
            gradient_x[i, j] = np.sum(sobel_x * region)
            gradient_y[i, j] = np.sum(sobel_y * region)

    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)

    return gradient_x, gradient_y, gradient_magnitude



def final_harris_image(img, window_size, k, threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # Get the height and width of the image
    height = img.shape[0]
    width = img.shape[1]

    # Initialize the response matrix R
    matrix_R = np.zeros((height, width))

    # Step 1: Compute the x and y gradients of the image using the Sobel operator
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: Compute the products of derivatives
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    # Define the offset for the window size
    offset = int(window_size / 2)

    # Step 3: Compute the sum of products of derivatives for each pixel
    print("Finding Corners...")
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(dx2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            # Step 4: Define the matrix H for each pixel
            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            # Step 5: Compute the response function R for each pixel
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det - k * (tr ** 2)
            matrix_R[y - offset, x - offset] = R

    # Step 6: Normalize the response matrix R and apply a threshold
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrix_R[y, x]
            if value > threshold:
                # Draw a circle at each corner point
                cv2.circle(img, (x, y), 3, (0, 255, 0))







###################################################################



def load_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return img


def canny_edge_detector(image, kernel_size, sigma, lowThresholdRatio, highThresholdRatio):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('image: ')
    print('image passed')
    print(f"sigma {sigma}, kernel {kernel_size} lowThresholdRatio {lowThresholdRatio} highThresholdRatio {highThresholdRatio}")
    kernel = gaussian_kernel(kernel_size, sigma)
    print('kernel:')
    print(kernel)
    edges = convolve(image, kernel)
    G, theta = sobel_filters(edges)
    non_max_suppression_image = non_max_suppression(G, theta)
    thresholded_image = threshold(non_max_suppression_image, lowThresholdRatio, highThresholdRatio)
    hysteresis_image = hysteresis(thresholded_image)
    return hysteresis_image

def gaussian_kernel(size, sigma=1):
    """Function to generate a Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def convolve(image, kernel):
    """Function to apply convolution."""
    m, n = kernel.shape
    y, x = image.shape
    print ('coonvol')
    y = y - m + 1
    x = x - m + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i + m, j:j + m] * kernel)
    return new_image


def sobel_filters(img):
    """Function to apply Sobel filters."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta


def non_max_suppression(img, D):
    """Function for non-maximum suppression."""
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass
    return Z


def threshold(img: object, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    """Function to apply threshold."""
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    res = np.zeros(img.shape, dtype=np.int32)
    weak = np.int32(75)
    strong = np.int32(255)
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res


def hysteresis(img):
    """Function for hysteresis."""
    M, N = img.shape
    weak = 75
    strong = 255
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img




def hough_lines_draw(img, img2, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta)
        b = np.sin(theta)
        pt0 = rho * np.array([a,b])
        pt1 = tuple((pt0 + 1000 * np.array([-b, a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b, a])).astype(int))
        if pt1[0] < 0:
            cv2.line(img, pt1, pt2, (255, 255, 0), 3)
        else:
            print(a, b)
            cv2.line(img2, pt1, pt2, (255, 255, 255), 3)

    return img


def hough_lines_draw_final(img, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta)
        b = np.sin(theta)
        pt0 = rho * np.array([a, b])
        pt1 = tuple((pt0 + 1000 * np.array([-b, a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b, a])).astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 255), 2)
    
    return img


def hough_lines_acc(img, rho_res=1, thetas = np.arange(-90, 90, 1)):
    rho_max = int(np.linalg.norm(img.shape-np.array([1, 1]), 2))
    rhos = np.arange(-rho_max, rho_max, rho_res)
    thetas -= min(min(thetas),0)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    yis, xis = np.nonzero(img)  # use only edge points
    for idx in range(len(xis)):
        x = xis[idx]
        y = yis[idx]
        temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
        temp_rhos = temp_rhos / rho_res + rho_max
        m, n = accumulator.shape
        valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
        temp_rhos = temp_rhos[valid_idxs]
        temp_thetas = thetas[valid_idxs]
        c = np.stack([temp_rhos,temp_thetas], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _, idxs, counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs].astype(np.uint)
        accumulator[uc[:, 0], uc[:, 1]] += counts.astype(np.uint)
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, thetas, rhos


def clip(idx):
    return int(max(idx, 0))


def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((numpeaks, 2), dtype=np.uint64)
    temp_H = H.copy()
    for i in range(numpeaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(temp_H)  # find maximum peak
        if max_val > threshold:
            peaks[i] = max_loc
            (c, r) = max_loc
            t = nhood_size//2.0
            temp_H[clip(r-t):int(r+t+1), clip(c-t):int(c+t+1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:, ::-1]





def final_hough_image(img, numpeaks, threshold, nhood_size):
    
    image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image2 = img
    edge_img =canny_edge_detector(img, 5, 6, 0.01, 0.09)

    cv2.imwrite("building.jpg", edge_img)
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, numpeaks=10, threshold=135, nhood_size=10)
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, numpeaks, threshold, nhood_size)

    # color_img1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    # color_img2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # res_img = hough_lines_draw(image1, color_img2, peaks, rhos, thetas)
    final = hough_lines_draw_final(image1, peaks, rhos, thetas)
    
    return final



class ImageDisplayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Display")
        self.resize(400, 300)  # Initial size
        
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)  # Center the image
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Fill available space
        self.setCentralWidget(self.label)
        
    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.resize(800, 600)  # Initial size
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.image_display_window1 = ImageDisplayWindow()
        self.image_display_window2 = ImageDisplayWindow()

        # Create a new horizontal layout for the images
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.image_display_window1)
        self.image_layout.addWidget(self.image_display_window2)

        # Add the image layout to the main layout
        self.layout.addLayout(self.image_layout)

        # Initialize file_path as an instance variable
        self.file_path = ""
        

        
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["Canny Edge Detector", "Hough Line Detector", "Harris Corner Detector"])
        self.algorithm_selector.currentIndexChanged.connect(self.update_ui)
        self.layout.addWidget(self.algorithm_selector)
        
        self.sigma_slider_label = QLabel("Sigma:")
        self.layout.addWidget(self.sigma_slider_label)
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(10)
        self.sigma_slider.setValue(1)
        self.layout.addWidget(self.sigma_slider)
        self.sigma_slider.valueChanged.connect(self.update_slider_value)
        self.sigma_slider_value_label = QLabel(f"Selected Value: {self.sigma_slider.value()}")
        self.layout.addWidget(self.sigma_slider_value_label)
        sigma_value = self.sigma_slider.value() / 100 
        print('sigma_value:')
        print(sigma_value)

        self.resolution_slider_label = QLabel("resolution:")
        self.layout.addWidget(self.resolution_slider_label)
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setMinimum(1)
        self.resolution_slider.setMaximum(180)
        self.resolution_slider.setValue(1)
        self.layout.addWidget(self.resolution_slider)
        self.resolution_slider.valueChanged.connect(self.update_slider_value)
        self.resolution_slider_value_label = QLabel(f"Selected resolution Value: {self.resolution_slider.value()}")
        self.layout.addWidget(self.resolution_slider_value_label)
        resolution_value = self.resolution_slider.value() / 100 
        print('resolution_value:')
        print(resolution_value)

        #slider for num of lines
        self.num_of_line_slider_label = QLabel("number of line:")
        self.layout.addWidget(self.num_of_line_slider_label)
        self.num_of_line_slider = QSlider(Qt.Horizontal)
        self.num_of_line_slider.setMinimum(1)
        self.num_of_line_slider.setMaximum(40)
        self.num_of_line_slider.setValue(1)
        self.layout.addWidget(self.num_of_line_slider)
        self.num_of_line_slider.valueChanged.connect(self.update_slider_value)
        self.num_of_line_slider_value_label = QLabel(f"Selected number of line Value: {self.num_of_line_slider.value()}")
        self.layout.addWidget(self.num_of_line_slider_value_label)
        num_of_line_value = self.num_of_line_slider.value() / 100 
        print('num_of_line:')
        print(num_of_line_value)

        #nhood resolution in hough transform
        self.nhood_slider_label = QLabel("low resolution:")
        self.layout.addWidget(self.nhood_slider_label)
        self.nhood_slider = QSlider(Qt.Horizontal)
        self.nhood_slider.setMinimum(1)
        self.nhood_slider.setMaximum(40)
        self.nhood_slider.setValue(1)
        self.layout.addWidget(self.nhood_slider)
        self.nhood_slider.valueChanged.connect(self.update_slider_value)
        self.nhood_slider_value_label = QLabel(f"Selected low resolution Value: {self.nhood_slider.value()}")
        self.layout.addWidget(self.nhood_slider_value_label)
        nhood_value = self.nhood_slider.value() / 100 
        print('nhood:')
        print(nhood_value)
        
# kernel label
        self.size_input_label = QLabel("Kernel Size:")
        self.layout.addWidget(self.size_input_label)

        self.size_input = QComboBox()
        self.size_input.addItems(["3", "5", "7", "9", "11"])
        self.layout.addWidget(self.size_input)
    


        

        self.threshold_label = QLabel("Thresholds (T1, T2):")
        self.layout.addWidget(self.threshold_label)
        
        self.threshold_slider1 = QSlider(Qt.Horizontal)
        self.threshold_slider1.setMinimum(0)
        self.threshold_slider1.setMaximum(100)
        self.threshold_slider1.setValue(50)
        self.layout.addWidget(self.threshold_slider1)
        self.threshold_slider1.valueChanged.connect(self.update_slider_value)
        self.threshold_slider1_value_label = QLabel(f"Selected Value: {self.threshold_slider1.value() / 100}")
        self.layout.addWidget(self.threshold_slider1_value_label)
        t1_value = self.threshold_slider1.value() / 100
        print('t1_value:')
        print(t1_value)
        


        self.threshold_slider2 = QSlider(Qt.Horizontal)
        self.threshold_slider2.setMinimum(0)
        self.threshold_slider2.setMaximum(100)
        self.threshold_slider2.setValue(100)
        self.layout.addWidget(self.threshold_slider2)
        self.threshold_slider2.valueChanged.connect(self.update_slider_value)
        self.threshold_slider2_value_label = QLabel(f"Selected Value: {self.threshold_slider2.value() / 1000}")
        self.layout.addWidget(self.threshold_slider2_value_label)
        t2_value = self.threshold_slider2.value() / 1000
        print('t2_value:')
        print(t2_value)

        self.apply_button = QPushButton("Apply")
        self.layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_filter)
        
        self.browse_button1 = QPushButton("Browse Image 1")
        self.browse_button1.clicked.connect(self.browse_image1)
        self.layout.addWidget(self.browse_button1)
        image1 = self.browse_image1
        print('image1:')
        print(self.browse_image1())



    def apply_filter(self):
        index = self.algorithm_selector.currentIndex()
        if index == 0:  # Canny Edge Detector
            sigma = self.sigma_slider.value()
            kernel_size = int(self.size_input.currentText())
            t1 = self.threshold_slider1.value() / 1000
            t2 = self.threshold_slider2.value() / 1000
            print('sigma:')
            print(sigma)
            print('kernel_size:')
            print(kernel_size)
            print('t1:')
            print(t1)
            print('t2:')
            print(t2)
            
            
            # print('image_path:')
            # print(image_path)
            
            print('file_path inside second window:')
            print(self.file_path)
            img = load_image(self.file_path) 
            img = canny_edge_detector(img,kernel_size, sigma, t1, t2)
            print(img)
            cv2.imwrite("saved_image.jpg", img)
            print("Image saved successfully.")



            

        

            # Load the image using OpenCV
            # img_gray = cv2.imread(image_path)
            self.image_display_window2.set_image("saved_image.jpg")
            # gray_image = cv2.cvtColor(self.image_display_window1, cv2.COLOR_BGR2GRAY)
            # self.image_display_window2.imshow('Grayscale Image', gray_image)
            
        elif index == 1:
            resolution = self.resolution_slider.value()
            low_resolution = self.nhood_slider.value()
            img = load_image(self.file_path) 
            num_of_lines = self.num_of_line_slider.value()
            img = final_hough_image(img, num_of_lines, resolution, low_resolution)
            
            
            cv2.imwrite("entasarna.jpg", img)
            self.image_display_window2.set_image("entasarna.jpg")

        elif index == 2:
            img = load_image(self.file_path)
            final_harris_image(img, 5, 0.04, 0.30)  # Change this path to one that will lead to your image
            cv2.imwrite("entasarna.jpg", img)
            self.image_display_window2.set_image("entasarna.jpg")
            
               

    
    def update_ui(self, index):
        if index == 0:  # Canny Edge Detector
            self.sigma_slider_label.show()
            self.sigma_slider.show()
            self.sigma_slider_value_label.show()
            self.size_input_label.show()
            self.size_input.show()
            self.threshold_label.show()
            self.threshold_slider1.show()
            self.threshold_slider2.show()
            self.threshold_slider1_value_label.show()
            self.threshold_slider2_value_label.show()


            self.resolution_slider_label.hide()
            self.resolution_slider.hide()
            self.resolution_slider_value_label.hide()

           
            self.nhood_slider_label.hide()
            self.nhood_slider.hide()
            self.nhood_slider_value_label.hide()
           

            self.num_of_line_slider_label.hide()
            self.num_of_line_slider.hide()
            self.num_of_line_slider_value_label.hide()
        if index == 1:  # Your additional logic here
            # Show the resolution slider and number of lines slider
            self.resolution_slider_label.show()
            self.resolution_slider.show()
            self.resolution_slider_value_label.show()
            self.nhood_slider_label.show()
            self.nhood_slider.show()
            self.nhood_slider_value_label.show()
           
            self.sigma_slider_label.hide()
            self.sigma_slider.hide()
            self.sigma_slider_value_label.hide()
            self.size_input_label.hide()
            self.size_input.hide()
            self.threshold_label.hide()
            self.threshold_slider1.hide()
            self.threshold_slider2.hide()
            self.threshold_slider1_value_label.hide()
            self.threshold_slider2_value_label.hide()

        else:
            # Hide all sliders and labels when index is neither 0 nor 1
            self.sigma_slider_label.hide()
            self.sigma_slider.hide()
            self.sigma_slider_value_label.hide()
            self.size_input_label.hide()
            self.size_input.hide()
            self.threshold_label.hide()
            self.threshold_slider1.hide()
            self.threshold_slider2.hide()
            self.threshold_slider1_value_label.hide()
            self.threshold_slider2_value_label.hide()
            self.resolution_slider_label.hide()
            self.resolution_slider.hide()
            self.resolution_slider_value_label.hide()
            self.num_of_line_slider_label.hide()
            self.num_of_line_slider.hide()
            self.num_of_line_slider_value_label.hide()

            self.nhood_slider_label.hide()
            self.nhood_slider.hide()
            self.nhood_slider_value_label.hide()

    
    def browse_image1(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            print('file_path:') 
            print(self.file_path)    
            self.image_display_window1.set_image(self.file_path)
    
  
    def update_slider_value(self):
        sender = self.sender()
        if sender == self.sigma_slider:
            self.sigma_slider_value_label.setText(f"Selected Value: {sender.value()}")
        elif sender == self.threshold_slider1:
            self.threshold_slider1_value_label.setText(f"Selected Value: {sender.value() / 100}")
        elif sender == self.threshold_slider2:
            self.threshold_slider2_value_label.setText(f"Selected Value: {sender.value() / 100}")
        elif sender == self.resolution_slider:
            self.resolution_slider_value_label.setText(f"Selected Value: {sender.value() }")    
        elif sender == self.num_of_line_slider:
            self.num_of_line_slider_value_label.setText(f"Selected Value: {sender.value() }")    
        elif sender == self.nhood_slider:
            self.num_of_line_slider_value_label.setText(f"Selected Value: {sender.value() }")    

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
