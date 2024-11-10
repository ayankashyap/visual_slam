import numpy as np
import cv2

def imshow(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawmatches(img1, pts1, img2, pts2):
    pts1 = pts1.astype(int)
    pts2 = pts2.astype(int)
    
    # Get dimensions of the input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create an empty canvas large enough to hold both images side by side
    matched_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    matched_img[:h1, :w1] = img1
    matched_img[:h2, w1:w1 + w2] = img2
    
    # Define colors for the points and lines
    color_circle = (0, 255, 0)  # Green for points
    color_line = (0, 255, 0)    # Green for lines

    # Draw circles on the matched points and lines connecting them
    for pt1, pt2 in zip(pts1, pts2):
        # pt1 in img1, pt2 in img2, but shifted horizontally by w1 for img2
        pt2_shifted = (pt2[0] + w1, pt2[1])
        
        # Draw circles on each point
        cv2.circle(matched_img, tuple(pt1), 5, color_circle, -1)
        cv2.circle(matched_img, pt2_shifted, 5, color_circle, -1)
        
        # Draw a line connecting the points
        cv2.line(matched_img, tuple(pt1), pt2_shifted, color_line, 1)

    return matched_img

def verify_F_constraint(pts1_homo, pts2_homo, F):
    # Convert points to homogeneous coordinates
    # Test x2^T F x1 = 0 for first few points
    for i in range(min(5, len(pts1_homo))):  # Show first 5 points
        x1 = pts1_homo[i]
        x2 = pts2_homo[i]
        constraint_value = x2.T @ F @ x1
        print(f"Point pair {i}: x2^T F x1 = {constraint_value:.2e}")
        
    # Show mean for all points
    constraints = [pts2_homo[i].T @ F @ pts1_homo[i] for i in range(len(pts1_homo))]
    print(f"\nMean absolute constraint value: {np.mean(np.abs(constraints)):.2e}")

def draw_epipolar_lines(img1, pts1, img2, pts2, F, circle_radius=10, line_radius=10):
    # assume homogenous points
    el_img2 = (F @ pts1.T).T
    el_img1 = (F.T @ pts2.T).T

    # combined_image
    h1, w1, c = img1.shape
    h2, w2, c = img2.shape

    combined_img = np.zeros((max(h1, h2), w1 + w2, c), dtype=np.uint8)
    l_img1 = img1.copy()
    l_img2 = img2.copy()
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    for p1, p2, e2, e1 in zip(pts1, pts2, el_img2, el_img1):
        # draw epipolar line in img1
        color = np.random.randint(0, 255, (3)).tolist()
        a,b,c = e1.reshape(-1)
        (x1, y1) = (int(-c/ a), 0)
        (x2, y2) = ( int((-b*max(h1,h2) - c) / a) , max(h1,h2)) 

        cv2.circle(l_img1, (p1[0], p1[1]), circle_radius, color, -1)
        cv2.line(l_img1, (x1, y1), (x2, y2), color, line_radius)
        
        a,b,c = e2.reshape(-1)
        (x1, y1) = (int(-c/ a), 0)
        (x2, y2) = ( int((-b*max(h1,h2) - c) / a) , max(h1,h2)) 

        cv2.circle(l_img2, (p2[0], p2[1]), circle_radius, color, -1)
        cv2.line(l_img2, (x1, y1), (x2, y2), color, line_radius)

    combined_img[:h1, :w1] = l_img1
    combined_img[:h2, w1:w1+w2] = l_img2

    return combined_img



