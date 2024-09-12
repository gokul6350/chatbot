import cv2
import pytesseract
from pytesseract import Output
import re

# Load image
image = cv2.imread('/home/gokul/Documents/chatbot/pdf/78.png')

# Perform OCR on the image
d = pytesseract.image_to_data(image, output_type=Output.DICT)

# List of sentences to highlight
highlight_sentences = [
    "Comprehensive Review on Alzheimer's Disease: Causes and Treatment"
]

def preprocess_text(text):
    # Remove extra whitespace and convert to lowercase
    return ' '.join(text.lower().split())

# Preprocess the sentences to highlight
processed_highlight_sentences = [preprocess_text(s) for s in highlight_sentences]

# Process all text from the image
full_text = ' '.join(d['text'])
processed_full_text = preprocess_text(full_text)

# Find matches and their positions
matches = []
for sentence in processed_highlight_sentences:
    for match in re.finditer(re.escape(sentence), processed_full_text):
        matches.append((match.start(), match.end()))

# Highlight matching sentences
for start, end in matches:
    # Find the corresponding word indices
    start_word = 0
    end_word = 0
    current_length = 0
    for i, word in enumerate(d['text']):
        word_length = len(preprocess_text(word))
        if current_length <= start < current_length + word_length:
            start_word = i
        if current_length < end <= current_length + word_length:
            end_word = i
            break
        current_length += word_length + 1  # +1 for space

    # Calculate bounding box
    x = min(d['left'][i] for i in range(start_word, end_word + 1))
    y = min(d['top'][i] for i in range(start_word, end_word + 1))
    right = max(d['left'][i] + d['width'][i] for i in range(start_word, end_word + 1))
    bottom = max(d['top'][i] + d['height'][i] for i in range(start_word, end_word + 1))

    # Highlight the sentence
    cv2.rectangle(image, (x, y), (right, bottom), (0, 255, 0), 2)  # Green box

# Save or display the image
cv2.imwrite('highlighted_image.jpg', image)
cv2.imshow('Highlighted Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
