import fitz  # PyMuPDF
import difflib
import os

# ANSI color codes (for console output)
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def is_close_match(a, b, threshold=0.8):
    # Normalize and compare strings based on a similarity threshold
    normalized_a = ''.join(e.lower() for e in a if e.isalnum() or e.isspace())
    normalized_b = ''.join(e.lower() for e in b if e.isalnum() or e.isspace())
    similarity = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()
    print(f"Comparing:\n'{a}'\nwith:\n'{b}'\nSimilarity: {similarity:.2f}")
    return similarity > threshold

def annotate_key_sentences(pdf_path, output_pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Processing PDF: {pdf_path}")

    # Key sentences identified from the document to highlight
    key_sentences = [
        "Alzheimer’s disease (AD) is a disorder that causes degeneration of the cells in the brain and is the main cause of dementia.",
        "AD is considered a multifactorial disease: two main hypotheses were proposed as a cause for AD, cholinergic and amyloid hypotheses.",
        "Currently, there are only two classes of approved drugs to treat AD, including inhibitors to cholinesterase enzyme and antagonists to N-methyl d-aspartate (NMDA).",
        "This review discusses currently available drugs and future theories for the development of new therapies for AD.",
        "A patient suspected to have AD should undergo several tests, including neurological examination, magnetic resonance imaging (MRI).",
        "In 1984, The National Institute of Neurological and Communicative Disorders and Stroke (NINCDS) and the Alzheimer’s Disease and Related Disorders Association (ADRDA) formed a work group to establish a clinical diagnostic’s criteria for Alzheimer’s disease.",
        "Neurofibrillary tangles and amyloid plaques are identified as primary markers.",
        "There is no cure for Alzheimer’s disease, although there are available treatments that just improve the symptoms.",
        "The research is focusing on understanding AD pathology by targeting several mechanisms."
    ]

    found_sentences = []
    not_found_sentences = key_sentences.copy()

    for page_num, page in enumerate(doc):
        print(f"\nProcessing page {page_num + 1}")
        page_text = page.get_text("text")
        for sentence in key_sentences[:]:  # Iterate over a copy of the list
            if sentence in not_found_sentences:
                matches = [line for line in page_text.splitlines() if is_close_match(sentence, line)]
                if matches:
                    match = matches[0]  # Take only the first match
                    found_sentences.append((sentence, f"Page {page_num + 1}: {match}"))
                    not_found_sentences.remove(sentence)
                    key_sentences.remove(sentence)  # Remove the sentence from future searches
                    
                    # Highlight the matched text in the PDF
                    text_instances = page.search_for(match)
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

    # Save the annotated PDF
    doc.save(output_pdf_path)
    doc.close()

    # Print results to console (with colors)
    print("\nSentences found and highlighted:")
    for original, found in found_sentences:
        print(f"{BLUE}Original: {original}{RESET}")
        print(f"{GREEN}Found: {found}{RESET}")
        print()

    print("\nSentences not found:")
    for sentence in not_found_sentences:
        print(f"{RED}{sentence}{RESET}")

    print(f"\nTotal sentences found and highlighted: {len(found_sentences)}")
    print(f"Total sentences not found: {len(not_found_sentences)}")
    print(f"Annotated PDF saved as: {output_pdf_path}")

# File paths
pdf_path = "/home/gokul/Documents/chatbot/pdf/pdf1.py.pdf"
output_dir = os.path.dirname(pdf_path)
output_filename = "Annotated_Document.pdf"
output_pdf_path = os.path.join(output_dir, output_filename)

annotate_key_sentences(pdf_path, output_pdf_path)
