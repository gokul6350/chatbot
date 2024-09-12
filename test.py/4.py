import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np

# Path to your PDF file
pdf_path = '/home/gokul/Documents/chatbot/pdf/test1.pdf'
# Path to Tesseract executable
#pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract.exe'

# Convert PDF to a list of images
pages = convert_from_path(pdf_path, 300)

# Process each page
for page_number, image in enumerate(pages):
    # Convert PIL image to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Use Tesseract to detect text and their bounding boxes
    d = pytesseract.image_to_data(open_cv_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if d['text'][i].lower().strip() == "comprehensive review on alzheimer’s disease":
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # Highlight the text
            cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save or display the result
    cv2.imshow(f'Highlighted Page {page_number+1}', open_cv_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
