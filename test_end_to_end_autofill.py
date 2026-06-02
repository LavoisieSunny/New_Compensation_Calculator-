import unittest
import sys
import os

# Append project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.parser_heuristics import format_suggestions_for_calculator

class TestEndToEndAutofill(unittest.TestCase):
    def test_autofill_flow_injury_case(self):
        # Simulate LLM Extraction data structure
        llm_extracted_data = {
            "case_type": "injury",
            "name": "Kumari Sneha Chakravarti",
            "age": 6,
            "medical_expenses": 84600,
            "award_amount": 106600
        }
        
        # Run conversion to calculator-ready format
        formatted = format_suggestions_for_calculator(llm_extracted_data)
        
        # Assertions:
        self.assertEqual(formatted["case_type"], "injury", "Injury remains injury")
        self.assertNotEqual(formatted["case_type"], "death", "No death conversion")
        
        fields = formatted["fields"]
        self.assertEqual(fields.get("injured_name"), "Kumari Sneha Chakravarti", "Name mapped correctly")
        self.assertEqual(fields.get("medical_expenses"), 84600, "Medical expenses preserved")
        
        # Award amount maps to total_compensation (or award_amount) in the suggestions payload
        self.assertEqual(formatted.get("total_compensation"), 106600, "Award amount preserved")

if __name__ == "__main__":
    unittest.main()
