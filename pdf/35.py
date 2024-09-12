import cv2
import pytesseract
from PIL import Image, ImageDraw

def find_and_highlight_sentence(image_path, input_sentence):
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Perform OCR
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Initialize variables
    n_boxes = len(ocr_data['text'])
    detected_sentences = []
    current_sentence = []
    bounding_boxes = []

    # Group words into sentences
    for i in range(n_boxes):
        word = ocr_data['text'][i].strip()
        if word:
            current_sentence.append(word)
            bounding_boxes.append((ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]))

            # Check for punctuation or line breaks
            if word[-1] in '.!?':
                detected_sentences.append((' '.join(current_sentence), bounding_boxes))
                current_sentence = []
                bounding_boxes = []
    
    # Handle the last sentence if not punctuated
    if current_sentence:
        detected_sentences.append((' '.join(current_sentence), bounding_boxes))

    # Match the input sentence
    for sentence, boxes in detected_sentences:
        if sentence.strip() == input_sentence.strip():
            # Highlight the sentence
            for (x, y, w, h) in boxes:
                draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
            break

    # Save or show the image
    highlighted_image_path = 'highlighted_sentence_image.png'
    image.save(highlighted_image_path)
    image.show()

    return highlighted_image_path

# Example usage
image_path = '23.png'
input_sentence = 'Disease: Causes and Treatment'
highlighted_image_path = find_and_highlight_sentence(image_path, input_sentence)
