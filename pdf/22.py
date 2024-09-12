import cv2
import numpy as np  # Add this import at the top of your file
import pytesseract
import fitz  # PyMuPDF
import difflib
import os

# Ensure Tesseract is installed and available in your PATH
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update as necessary

def is_close_match(a, b, threshold=0.8):
    normalized_a = ''.join(e.lower() for e in a if e.isalnum() or e.isspace())
    normalized_b = ''.join(e.lower() for e in b if e.isalnum() or e.isspace())
    similarity = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()
    return similarity > threshold

def detect_text_with_boxes(image):
    # Use Tesseract to extract text and bounding boxes from the image
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return data

def highlight_sentences_in_pdf(pdf_path, output_pdf_path, key_sentences):
    doc = fitz.open(pdf_path)
    
    found_sentences = []
    not_found_sentences = key_sentences.copy()

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        
        # Convert pixmap to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.alpha:
            # If the image has an alpha channel, convert from RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Perform OCR on the image with bounding boxes
        ocr_data = detect_text_with_boxes(img)

        # Create a copy of the image for drawing
        img_with_highlights = img.copy()
        found_on_page = False

        # Draw bounding boxes for all detected text
        for sentence in key_sentences[:]:
            if sentence in not_found_sentences:
                matches = [line for line in ocr_data['text'] if is_close_match(sentence, line)]
                if matches:
                    found_sentences.append((sentence, matches[0]))
                    not_found_sentences.remove(sentence)
                    key_sentences.remove(sentence)
                    found_on_page = True
                    
                    # Search for and highlight the detected text in the PDF
                    text_instances = page.search_for(matches[0])
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()
                    
                    # Highlight the matched sentence in red
                    for i, text in enumerate(ocr_data['text']):
                        if is_close_match(matches[0], text):
                            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                            cv2.rectangle(img_with_highlights, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(img_with_highlights, f"Found: {text[:30]}...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if found_on_page:
            # Display the image with all bounding boxes
            cv2.imshow(f"Highlighted Sentences on Page {page_num + 1}", img_with_highlights)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Save the annotated PDF
    doc.save(output_pdf_path)
    doc.close()

    print("Highlighting completed. Sentences found:", len(found_sentences))
    print("Sentences not found:", len(not_found_sentences))

# Key sentences to search for and highlight
key_sentences = [
    "Alzheimer’s disease (AD) is a disorder that causes degeneration of the cells in the brain and is the main cause of dementia.",
    "Alzheimer’s",
    # Add other key sentences
]

# File paths
pdf_path = "test1.pdf"
output_pdf_path = "Annotated_Output.pdf"

highlight_sentences_in_pdf(pdf_path, output_pdf_path, key_sentences)
