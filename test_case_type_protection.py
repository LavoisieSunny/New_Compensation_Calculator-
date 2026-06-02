import unittest

def apply_autofill_with_protection(user_case_type, llm_case_type):
    """
    Simulates the workstation case type overwrite protection logic.
    If the user has manually selected a case type, it cannot be overwritten by LLM suggestions.
    """
    overwrite_blocked = False
    final_case_type = user_case_type
    
    if user_case_type and user_case_type != "":
        if llm_case_type and llm_case_type != user_case_type:
            overwrite_blocked = True
        final_case_type = user_case_type
    else:
        final_case_type = llm_case_type or "death"
        
    return final_case_type, overwrite_blocked

class TestCaseTypeProtection(unittest.TestCase):
    def test_overwrite_protection_injury_selected(self):
        # Scenario: User manually selects: case_type = "injury"
        user_case_type = "injury"
        llm_suggestions = {
            "case_type": "death",
            "name": "Sneha"
        }
        
        final_case_type, overwrite_blocked = apply_autofill_with_protection(
            user_case_type, llm_suggestions.get("case_type")
        )
        
        # Assertions
        self.assertEqual(final_case_type, "injury", "User selection must be preserved")
        self.assertTrue(overwrite_blocked, "LLM overwrite must be blocked")
        self.assertNotEqual(final_case_type, "death", "Injury workflow remains active")

    def test_overwrite_protection_death_selected(self):
        # Scenario: User manually selects: case_type = "death"
        user_case_type = "death"
        llm_suggestions = {
            "case_type": "injury",
            "name": "Amit"
        }
        
        final_case_type, overwrite_blocked = apply_autofill_with_protection(
            user_case_type, llm_suggestions.get("case_type")
        )
        
        # Assertions
        self.assertEqual(final_case_type, "death", "User selection must be preserved")
        self.assertTrue(overwrite_blocked, "LLM overwrite must be blocked")
        self.assertNotEqual(final_case_type, "injury", "Death workflow remains active")

    def test_no_protection_when_user_selection_empty(self):
        # Scenario: User has not selected a case type yet
        user_case_type = ""
        llm_suggestions = {
            "case_type": "injury",
            "name": "Sneha"
        }
        
        final_case_type, overwrite_blocked = apply_autofill_with_protection(
            user_case_type, llm_suggestions.get("case_type")
        )
        
        # Assertions
        self.assertEqual(final_case_type, "injury", "LLM suggestion should apply when empty")
        self.assertFalse(overwrite_blocked, "Overwrite is not blocked since user selection was empty")

if __name__ == "__main__":
    unittest.main()
