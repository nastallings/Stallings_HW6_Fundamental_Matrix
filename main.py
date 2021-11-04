import cv2
import matplotlib.pyplot as plt
import Class_Code.structure as structure
import Class_Code.processor as processor
import Class_Code.features as features
import numpy as np

def compare_images(img1, img2):
    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)

    height, width, ch = img1.shape
    intrinsic = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])

    return points1, points2, intrinsic

# Compute F
def Compute_F(img1, img2):
    # Find similar points using cv2 SIFT
    points1, points2, intrinsic = compare_images(img1, img2)
    # Compute Fundamental Matrix
    F = structure.compute_fundamental_normalized(points1, points2)
    # Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig = structure.plot_epipolar_lines(points1, points2, F, fig, ax1, ax2, True)
    return fig, ax1, ax2, F

if __name__ == "__main__":
    # Load Images
    img1 = cv2.imread('Model_House_Images/house.000.pgm')
    img2 = cv2.imread('Model_House_Images/house.001.pgm')
    fig, ax1, ax2, F = Compute_F(img1, img2)
    print(F)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

