import cv2
import numpy as np
import pygame
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw with Camera")
clock = pygame.time.Clock()

def detect_hand(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def draw_on_screen():
    cap = cv2.VideoCapture(0)
    drawing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        contours = detect_hand(frame)

      
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                pygame.draw.circle(screen, (255, 0, 0), (x + w//2, y + h//2), 5)
                drawing = True

        if not drawing:
            screen.fill((255, 255, 255))  # Clear screen if not drawing

        pygame.display.flip()
        drawing = False
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

      
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    draw_on_screen()
