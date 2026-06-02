import unittest
import sys
import os
import asyncio
from unittest.mock import patch

# Append project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main import chat_with_pdf, PDFChatRequest

class TestChatbotCompensationContext(unittest.TestCase):
    def test_chatbot_context_wording_and_values(self):
        # Scenario Input values:
        # - Awarded Compensation: 1,06,600
        # - Claim Amount: 7,50,000
        # - Enhancement Sought: 2,00,000
        # - Calculated Compensation: 270,000
        mock_payload = {
            "question": "What is the compensation amount?",
            "filename": "MA_609_2026.pdf",
            "ocr_text": "Amount Awarded Rs. 1,06,600. Claim before Tribunal: Rs. 7,50,000. Appeal valued at: Rs. 2,00,000",
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
            # 1. Chatbot prompt contains all four values in its context and instruction details
            self.assertIn("1,06,600", prompt, "Awarded compensation must be in prompt context")
            self.assertIn("7,50,000", prompt, "Claim amount must be in prompt context")
            self.assertIn("2,00,000", prompt, "Enhancement sought must be in prompt context")
            self.assertIn("270000", prompt, "Calculated compensation must be in prompt context")
            
            # 2. Verify NO "source of truth" wording is allowed (forbidden list matches exactly)
            self.assertIn("NEVER say: 'The deterministic calculator output supplied in the context is the absolute single source of truth' or 'the mathematical engine remains the ultimate source of truth'", prompt)
            self.assertIn("NEVER say: 'Calculated amount is the source of truth'", prompt)
            
            # 3. PDF award remains distinguishable from calculator output
            self.assertIn("awarded_compensation: Amount awarded by the Tribunal/Court", prompt)
            self.assertIn("calculated_compensation: Amount computed by the deterministic calculator", prompt)
            self.assertIn("The Tribunal award is a judicial fact, whereas the calculator output is a computed estimate.", prompt)

if __name__ == "__main__":
    unittest.main()
