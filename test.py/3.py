import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_image(pdf_path, page_number=0):
    # Convert PDF to image
    pages = convert_from_path(pdf_path, dpi=300)
    return np.array(pages[page_number])

def detect_text(image):
    # Perform OCR
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return data

def highlight_text(image, data):
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # Only consider text with confidence > 60%
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main(pdf_path):
    # Convert PDF to image
    image = pdf_to_image(pdf_path)
    
    # Detect text using OCR
    data = detect_text(image)
    
    # Highlight detected text areas
    annotated_image = highlight_text(image, data)
    
    # Save the annotated image
    cv2.imwrite('annotated_pdf.png', annotated_image)
    
    # Print detected text with locations
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            print(f"Text: {data['text'][i]}, Location: ({data['left'][i]}, {data['top'][i]})")

if __name__ == "__main__":
    pdf_path = "/home/gokul/Documents/chatbot/pdf/test1.pdf"
    main(pdf_path)