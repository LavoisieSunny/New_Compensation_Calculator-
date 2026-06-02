import unittest
import sys
import os
import asyncio
from unittest.mock import patch

# Append project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main import chat_with_pdf, PDFChatRequest

class TestCompensationSeparation(unittest.TestCase):
    def test_prompt_separates_compensation_fields(self):
        # Input representing the separate concepts
        mock_payload = {
            "question": "What is the compensation amount?",
            "filename": "MA_609_2026.pdf",
            "ocr_text": "Awarded Compensation: Rs. 1,06,600. Claim: Rs. 7,50,000. Enhancement: Rs. 2,00,000",
            "parsed_fields": {
                "case_type": "injury",
                "name": "Kumari Sneha",
                "age": 6,
                "monthly_income": 0,
                "disability": 0
            },
            "calculator_result": {
                "case_type": "injury",
                "final_amount": 270000,
                "total_compensation": 270000
            }
        }
        
        request = PDFChatRequest(**mock_payload)
        
        with patch("backend.llm_client.generate_response") as mock_generate, \
             patch("backend.vector_db.semantic_search_rag") as mock_rag:
             
            mock_generate.return_value = "Mocked Response"
            mock_rag.return_value = []
            
            # Run async function using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(chat_with_pdf(request))
            finally:
                loop.close()
            
            called_args, _ = mock_generate.call_args
            prompt = called_args[0]
            
            # Assertions:
            # 1. Verify all fields are preserved and referred separately in prompt guidelines
            self.assertIn("awarded_compensation", prompt, "awarded_compensation concept must be preserved")
            self.assertIn("claimed_compensation", prompt, "claimed_compensation concept must be preserved")
            self.assertIn("enhancement_sought", prompt, "enhancement_sought concept must be preserved")
            self.assertIn("calculated_compensation", prompt, "calculated_compensation concept must be preserved")
            
            # 2. Verify prompt specifically forbids merging or source of truth descriptions
            self.assertIn("NEVER say: 'Calculated amount is the source of truth'", prompt)
            self.assertIn("NEVER say: 'Compensation amount is \u20b9X'", prompt)
            self.assertIn("NEVER overwrite judicially awarded compensation with calculator output", prompt)
            
            # 3. Verify format demands
            self.assertIn("According to the PDF (Judicial Record)", prompt)
            self.assertIn("According to the Compensation Calculator", prompt)
            self.assertIn("Comparison", prompt)
            self.assertIn("Difference:", prompt)

if __name__ == "__main__":
    unittest.main()
