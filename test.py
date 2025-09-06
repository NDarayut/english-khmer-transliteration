import unittest
from inference import transliterate_text, transliterate_top_n, transliterate_with_dict

class TestKhmerTransliterator(unittest.TestCase):

    def test_transliterate_text_returns_string(self):
        """Test that transliterate_text returns a string for normal input"""
        result = transliterate_text("srolanh")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0, "Output should not be empty")

    def test_transliterate_text_empty_input(self):
        """Test that empty input returns empty string"""
        result = transliterate_text("")
        self.assertEqual(result, "")

    def test_transliterate_top_n_returns_list(self):
        """Test that transliterate_top_n returns a list of length n"""
        n = 3
        result = transliterate_top_n("srolanh", n=n)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), n)
        for item in result:
            self.assertIsInstance(item, str)

    def test_transliterate_with_dict_returns_list(self):
        """Test that transliterate_with_dict returns a list of valid Khmer words"""
        n = 5
        result = transliterate_with_dict("srolanh", n=n)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), n)
        for word in result:
            self.assertIsInstance(word, str)
            self.assertTrue(len(word) > 0, "Each transliterated word should not be empty")

    def test_transliterate_with_dict_empty_input(self):
        """Test that empty input returns empty list"""
        result = transliterate_with_dict("")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
