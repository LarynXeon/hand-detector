import cv2
import numpy as np

def create_bone_figure(hand_mask):
    bone_figure = np.zeros_like(hand_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(bone_figure, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(largest_contour[s][0])
                end = tuple(largest_contour[e][0])
                cv2.line(bone_figure, start, end, (255, 255, 255), thickness=5)
                
    return bone_figure

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Thresholding to extract hand region
        _, hand_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Invert the mask
        hand_mask = cv2.bitwise_not(hand_mask)
        
        # Create bone-like figure
        bone_figure = create_bone_figure(hand_mask)
        
        # Display the bone figure
        cv2.imshow('Bone Figure', bone_figure)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
