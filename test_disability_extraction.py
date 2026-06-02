import unittest
import sys
import os

# Append project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.parser_heuristics import parse_extracted_text

class TestDisabilityExtraction(unittest.TestCase):
    def test_no_disability_hallucination_on_enhancement_and_interest(self):
        # Scenario: OCR Text containing enhancement and interest percentages but no disability mentions
        ocr_text = (
            "Amount Awarded Rs. 106600\n"
            "Enhancement sought 50%\n"
            "Interest 6%"
        )
        
        text_lines = ocr_text.split("\n")
        suggestions = parse_extracted_text(text_lines)
        
        disability = suggestions.get("disability")
        
        # Assertions
        self.assertTrue(disability is None or disability == "", "Enhancement percentage and interest must not be mapped to disability")

    def test_valid_permanent_disability_extraction(self):
        # Scenario: Valid permanent disability declaration
        ocr_text = (
            "Claimant suffered a fracture in accident\n"
            "Medical Board assessed Permanent Disability: 50%\n"
            "Monthly income was Rs 20,000\n"
        )
        
        text_lines = ocr_text.split("\n")
        suggestions = parse_extracted_text(text_lines)
        
        disability = suggestions.get("disability")
        
        # Assertions
        self.assertEqual(disability, 50.0, "Permanent Disability must be correctly parsed as 50")

    def test_valid_functional_disability_extraction(self):
        # Scenario: Valid functional disability declaration
        ocr_text = (
            "Claimant suffered severe head injury\n"
            "Functional Disability: 50%\n"
            "Annual income is Rs. 1,20,000\n"
        )
        
        text_lines = ocr_text.split("\n")
        suggestions = parse_extracted_text(text_lines)
        
        disability = suggestions.get("disability")
        
        # Assertions
        self.assertEqual(disability, 50.0, "Functional Disability must be correctly parsed as 50")

if __name__ == "__main__":
    unittest.main()
