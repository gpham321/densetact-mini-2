import cv2
import numpy as np
import time

def detect_dark_circular_edge(frame, debug=False):
    """Detect the dark circular edge once and return center and radius"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Use edge detection to find the dark border
    edges = cv2.Canny(blurred, 30, 100)
    
    # Use HoughCircles to detect the circular edge
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,              # Resolution ratio
        minDist=200,       # Minimum distance between circle centers
        param1=50,         # Upper threshold for edge detection
        param2=25,         # Accumulator threshold for center detection
        minRadius=80,      # Minimum circle radius
        maxRadius=300      # Maximum circle radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # If multiple circles detected, choose the one closest to center
        if len(circles) > 1:
            height, width = frame.shape[:2]
            frame_center_x, frame_center_y = width // 2, height // 2
            
            # Find circle closest to frame center
            best_circle = min(circles, key=lambda c: 
                np.sqrt((c[0] - frame_center_x)**2 + (c[1] - frame_center_y)**2))
            center_x, center_y, radius = best_circle
        else:
            center_x, center_y, radius = circles[0]
        
        if debug:
            print(f"Dark edge detected: Center({center_x}, {center_y}), Radius: {radius}")
        
        return (center_x, center_y), radius, True
    
    if debug:
        print("No dark circular edge detected")
    
    return None, None, False

def create_fixed_circular_mask(frame, center, radius, buffer_factor=1.0):
    """Create circular mask using fixed center and radius - exactly on detected edge"""
    height, width = frame.shape[:2]
    center_x, center_y = center
    
    # Apply small buffer to radius
    buffered_radius = int(radius * buffer_factor)
    
    # Ensure circle stays within frame bounds
    max_radius = min(center_x, center_y, width - center_x, height - center_y)
    final_radius = min(buffered_radius, max_radius)
    
    # Create mask
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= final_radius
    
    return mask, final_radius

def apply_circular_crop(frame, mask, background_color=(0, 0, 0)):
    """Apply circular mask to frame"""
    masked_frame = frame.copy()
    masked_frame[~mask] = background_color
    return masked_frame

class AutoCircularCameraViewer:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.buffer_factor = 1.0  # Exactly on the detected edge
        self.debug_mode = False
        
        # Store detected circle parameters (detect once, use continuously)
        self.circle_center = None
        self.circle_radius = None
        self.circle_detected = False
        
        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            self.cap = None
            return
            
        # Test camera
        ret, test_frame = self.cap.read()
        if not ret:
            print(f"Error: Could not read from camera {camera_index}")
            self.cap.release()
            self.cap = None
            return
            
        height, width = test_frame.shape[:2]
        print(f"Camera {camera_index} initialized successfully!")
        print(f"Resolution: {width}x{height}")
        print("Ready to detect dark circular edge (mask will be exactly on the edge)")
        
    def detect_circle_once(self, frame):
        """Detect the dark circular edge once and store parameters"""
        center, radius, detected = detect_dark_circular_edge(frame, self.debug_mode)
        
        if detected:
            self.circle_center = center
            self.circle_radius = radius
            self.circle_detected = True
            print(f"Circle detected and locked: Center{center}, Radius: {radius}")
            return True
        else:
            print("No dark circular edge found - try adjusting lighting or position")
            return False
        
    def run(self):
        if self.cap is None:
            print("Camera not available")
            return
        
        print("\nOne-Time Circle Detection Controls:")
        print("'SPACE' - Detect dark circular edge (do this first!)")
        print("'r' - Re-detect circle (if needed)")
        print("'q' - Quit")
        print("'s' - Save frame")
        print("'d' - Toggle debug mode")
        print("'+' - Increase buffer size (outside edge)")
        print("'-' - Decrease buffer size (inside edge)")
        print()
        print("Press SPACE to detect the dark circular edge once!")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.prev_time)
            self.prev_time = current_time
            self.fps = fps
            
            height, width = frame.shape[:2]
            
            # Use stored circle parameters if available
            if self.circle_detected:
                # Create mask using fixed detected circle
                mask, final_radius = create_fixed_circular_mask(
                    frame, self.circle_center, self.circle_radius, self.buffer_factor
                )
                
                # Apply mask
                circular_frame = frame.copy()
                circular_frame[~mask] = (0, 0, 0)
                
                # No circle outline - removed all cv2.circle() calls
                
                # FPS display instead of "CIRCLE LOCKED"
                fps_text = f"FPS: {self.fps:.1f}"
                cv2.putText(circular_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            else:
                # No circle detected yet - show original frame
                circular_frame = frame.copy()
                
                # Status overlay
                status_text = "PRESS SPACE TO DETECT CIRCLE"
                cv2.putText(circular_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add controls reminder
            controls_text = "SPACE: Detect | R: Re-detect | +/-: Buffer | D: Debug | S: Save | Q: Quit"
            cv2.putText(circular_frame, controls_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('One-Time Circle Detection', circular_frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar - detect circle
                print("Detecting dark circular edge...")
                self.detect_circle_once(frame)
            elif key == ord('r'):  # Re-detect
                print("Re-detecting circle...")
                self.circle_detected = False
                self.detect_circle_once(frame)
            elif key == ord('s'):
                cv2.imwrite('fixed_circular_capture.png', circular_frame)
                print("Frame saved as 'fixed_circular_capture.png'")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.buffer_factor = min(1.1, self.buffer_factor + 0.01)
                print(f"Buffer factor: {self.buffer_factor:.2f}")
            elif key == ord('-'):
                self.buffer_factor = max(0.9, self.buffer_factor - 0.01)
                print(f"Buffer factor: {self.buffer_factor:.2f}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")

# Simple version for direct use
def simple_auto_detection(camera_index=1):
    """Simple version - detects circle once on first frame, then uses it"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print("Simple One-Time Circle Detection")
    print("Will detect dark circular edge on first frame automatically")
    print("Mask will be exactly on the detected edge")
    print("Press 'q' to quit, 's' to save, 'r' to re-detect")
    
    # Detection parameters
    circle_center = None
    circle_radius = None
    circle_detected = False
    buffer_factor = 1.0  # Exactly on the detected edge
    
    # FPS tracking
    prev_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        
        # Auto-detect on first frame or when requested
        if not circle_detected:
            center, radius, detected = detect_dark_circular_edge(frame)
            if detected:
                circle_center = center
                circle_radius = radius
                circle_detected = True
                print(f"Circle auto-detected: Center{center}, Radius: {radius}")
            else:
                print("No circle detected - showing original frame")
        
        # Apply mask if circle was detected
        if circle_detected:
            mask, final_radius = create_fixed_circular_mask(frame, circle_center, circle_radius, buffer_factor)
            circular_frame = frame.copy()
            circular_frame[~mask] = (0, 0, 0)
            
            # No circle outline - removed cv2.circle() calls
            
            # Show FPS instead of "CIRCLE LOCKED"
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(circular_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            circular_frame = frame.copy()
            status = "SEARCHING FOR CIRCLE..."
            cv2.putText(circular_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Simple Circle Detection', circular_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and circle_detected:
            cv2.imwrite('simple_fixed_capture.png', circular_frame)
            print("Frame saved!")
        elif key == ord('r'):  # Re-detect
            circle_detected = False
            print("Re-detecting circle...")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== One-Time Dark Edge Circle Detection ===")
    print("Detects the dark circular edge ONCE and uses it for all frames!")
    print()
    print("Perfect for microscopy specimens with dark borders")
    print("• Detects the dark edge boundary once")
    print("• Crops exactly on the detected edge") 
    print("• Uses same crop for all subsequent frames")
    print("• No continuous re-detection")
    print()
    print("Starting simple auto-detection mode...")
    print("Will detect dark circular edge on first frame automatically")
    print("Press 'q' to quit, 's' to save, 'r' to re-detect")
    print()
    
    simple_auto_detection()