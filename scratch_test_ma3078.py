import os
import sys
import pypdfium2 as pdfium

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.ocr import extract_digital_pdf_text, is_extracted_text_sparse, perform_ocr_on_scanned_pdf
from backend.parser_heuristics import parse_extracted_text

pdf_path = r"C:\Users\lavoi\Desktop\Miracle\pdfs\compensation\MA_3078_2021.pdf"
print("Checking digital extraction...")
digital_text = extract_digital_pdf_text(pdf_path)
print(f"Extracted {len(digital_text)} lines digitally.")

if is_extracted_text_sparse(digital_text):
    print("Digital text is sparse, running scanned OCR fallback...")
    # Render and scan all pages (or relevant pages)
    # Let's see how main backend does it:
    from backend.ocr import perform_ocr_on_scanned_pdf
    # Run OCR
    text_lines, ocr_debug = perform_ocr_on_scanned_pdf(pdf_path, scan_all_pages=False)
else:
    text_lines = digital_text

print("\n--- SUGGESTIONS ---")
suggestions = parse_extracted_text(text_lines)
import pprint
pprint.pprint(suggestions)

# Write raw text to file to inspect it
with open("ma3078_raw_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text_lines))
print("\nSaved text to ma3078_raw_text.txt")
