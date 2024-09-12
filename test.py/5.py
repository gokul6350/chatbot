import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Uncomment and set the path to the Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get a binary image
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Remove noise with a median blur
    cleaned_image = cv2.medianBlur(thresh_image, 3)
    return cleaned_image, image

def detect_text_in_image(image_path, target_text):
    # Preprocess image
    preprocessed_image, original_image = preprocess_image(image_path)
    
    # Use pytesseract to get text data
    details = pytesseract.image_to_data(preprocessed_image, output_type=Output.DICT)
    
    num_boxes = len(details['text'])
    found = False
    
    for sequence_number in range(num_boxes):
        text = details['text'][sequence_number].strip()
        if target_text.lower() in text.lower():  # Use partial matching
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            
            # Draw rectangle around the detected text
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_image, target_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            found = True

    if not found:
        print("Text not found in image.")
        return None

    # Save or display the result
    result_image_path = 'result_image.jpg'
    cv2.imwrite(result_image_path, original_image)
    cv2.imshow('Detected Text', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image_path

# Example usage
image_path = '/home/gokul/Documents/chatbot/pdf/highlighted_sentence_image.png'  # Replace with your image file path
target_text = "comprehensive review on alzheimer’s disease"
result_image_path = detect_text_in_image(image_path, target_text)
if result_image_path:
    print(f'Result image saved to {result_image_path}')
