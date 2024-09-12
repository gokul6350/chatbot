import cv2
import pytesseract
from pytesseract import Output
import time
import Levenshtein

# Load the image
image = cv2.imread('/home/gokul/Documents/chatbot/test.py/result_image.jpg')

# Perform OCR on the image
d = pytesseract.image_to_data(image, output_type=Output.DICT)

# Sentence to detect
sentence = ["alzheimer’s", "disease", "(ad)", "is", "a", "disorder", "that", "causes", "degeneration", "of", "the", "cells"]

# Convert sentence to lowercase for case-insensitive comparison
sentence = [word.lower() for word in sentence]

# Tolerance for y-coordinate to consider words on the same line
tolerance = 10

# Levenshtein distance threshold for word matching
levenshtein_threshold = 2

# Initialize variables
sentence_detected = False
n_boxes = len(d['text'])
i = 0

# Store coordinates of detected words
detected_words = []

while i < n_boxes:
    word = d['text'][i].strip().lower()
    
    for target_word in sentence:
        if Levenshtein.distance(word, target_word) <= levenshtein_threshold:  # Check if the word matches within the threshold
            x, y, w, h = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            detected_words.append((target_word, x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Highlight in green
            print(f"Found word: {word} (matched with {target_word}) at ({x}, {y}, {w}, {h})")
            break
    
    i += 1

# Check if detected words are on the same line
for i in range(len(detected_words) - len(sentence) + 1):
    match = True
    for j in range(len(sentence)):
        if detected_words[i + j][0] != sentence[j]:
            match = False
            break
        if j > 0 and abs(detected_words[i + j][2] - detected_words[i + j - 1][2]) > tolerance:
            match = False
            break
    
    if match:
        sentence_detected = True
        print("Sentence detected.")
        for k in range(len(sentence)):
            x, y, w, h = detected_words[i + k][1:]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Highlight in red
            print(f"Highlighting sentence word: {detected_words[i + k][0]} at ({x}, {y}, {w}, {h})")
        break

if not sentence_detected:
    print("Sentence not detected.")

# Display the image with a delay to visualize the process
cv2.imshow('Highlighted Image', image)
cv2.waitKey(2000)  # 2-second delay
cv2.destroyAllWindows()

# Save the result if needed
cv2.imwrite('highlighted_sentence.jpg', image)
