"""
Unit tests for data_cleaning.py module
Tests text processing, cleaning, and special format handling
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dags/common'))

from data_cleaning import TextCleaner


class TestTextCleaner(unittest.TestCase):
    """Test cases for TextCleaner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cleaner = TextCleaner()
    
    def test_clean_text_with_html(self):
        """Test HTML tag removal"""
        input_text = "<p>Hello <b>World</b>!</p>"
        expected = "Hello World!"
        result = self.cleaner.clean_text(input_text)
        self.assertEqual(result, expected)
    
    def test_clean_text_with_multiple_spaces(self):
        """Test multiple space normalization"""
        input_text = "Hello     World    Test"
        expected = "Hello World Test"
        result = self.cleaner.clean_text(input_text)
        self.assertEqual(result, expected)
    
    def test_clean_text_with_special_characters(self):
        """Test special character handling"""
        input_text = "Hello @ World # Test $ 123"
        expected = "Hello  World  Test  123"  
        result = self.cleaner.clean_text(input_text)
        self.assertEqual(result, expected)
    
    def test_clean_text_preserves_punctuation(self):
        """Test that basic punctuation is preserved"""
        input_text = "Hello, World! How are you? (Fine.)"
        expected = "Hello, World! How are you? (Fine.)"
        result = self.cleaner.clean_text(input_text)
        self.assertEqual(result, expected)
    
    def test_clean_text_with_none(self):
        """Test handling of None input"""
        result = self.cleaner.clean_text(None)
        self.assertEqual(result, "")
    
    def test_clean_text_with_empty_string(self):
        """Test handling of empty string"""
        result = self.cleaner.clean_text("")
        self.assertEqual(result, "")
    
    def test_clean_text_complex_html(self):
        """Test complex HTML with nested tags"""
        input_text = """
        <div class="article">
            <h1>Title</h1>
            <p>This is <span style="color:red">important</span> text.</p>
        </div>
        """
        result = self.cleaner.clean_text(input_text)
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertIn("Title", result)
        self.assertIn("important", result)
    
    def test_remove_latex_basic_commands(self):
        """Test basic LaTeX command removal"""
        input_text = r"\textbf{Important} \textit{text} here"
        expected = "Important text here"
        result = self.cleaner.remove_latex(input_text)
        self.assertEqual(result, expected)
    
    def test_remove_latex_citations(self):
        """Test LaTeX citation removal"""
        input_text = r"According to \cite{Smith2020} and \cite{Jones2021}"
        expected = "According to Smith2020 and Jones2021"
        result = self.cleaner.remove_latex(input_text)
        self.assertEqual(result, expected)
    
    def test_remove_latex_math_mode(self):
        """Test LaTeX math mode removal"""
        input_text = r"The equation $x^2 + y^2 = z^2$ is famous"
        expected = "The equation x^2 + y^2 = z^2 is famous"
        result = self.cleaner.remove_latex(input_text)
        self.assertEqual(result, expected)
    
    def test_remove_latex_with_none(self):
        """Test LaTeX removal with None input"""
        result = self.cleaner.remove_latex(None)
        self.assertEqual(result, "")
    
    def test_remove_latex_complex_commands(self):
        """Test complex LaTeX command removal"""
        input_text = r"\section{Introduction} \subsection{Background} \emph{emphasis}"
        result = self.cleaner.remove_latex(input_text)
        self.assertNotIn("\\section", result)
        self.assertNotIn("\\subsection", result)
        self.assertNotIn("\\emph", result)
        self.assertIn("emphasis", result)
    
    def test_edge_case_mixed_content(self):
        """Test mixed HTML and special characters"""
        input_text = "<p>Price: $100.00 @ 50% off!</p>"
        result = self.cleaner.clean_text(input_text)
        self.assertIn("Price:", result)
        self.assertIn("100.00", result)
        self.assertIn("50", result)
        self.assertNotIn("<p>", result)
    
    def test_unicode_handling(self):
        """Test Unicode character handling"""
        input_text = "Hello 世界 مرحبا мир"
        result = self.cleaner.clean_text(input_text)
        self.assertIn("Hello", result)
        # Unicode letters should be preserved
        self.assertIn("世界", result)
        self.assertIn("مرحبا", result)
        self.assertIn("мир", result)
    
    def test_whitespace_edge_cases(self):
        """Test various whitespace scenarios"""
        test_cases = [
            ("\n\nHello\n\nWorld\n\n", "Hello World"),
            ("\t\tHello\t\tWorld\t\t", "Hello World"),
            ("   Hello   World   ", "Hello World"),
            ("Hello\r\nWorld", "Hello World"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.cleaner.clean_text(input_text)
                self.assertEqual(result, expected)


class TestTextCleanerIntegration(unittest.TestCase):
    """Integration tests for TextCleaner with real-world examples"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cleaner = TextCleaner()
    
    def test_arxiv_abstract_cleaning(self):
        """Test cleaning of typical arXiv abstract"""
        input_text = r"""
        \textbf{Abstract:} We propose a novel approach to \emph{machine learning}
        using \cite{Author2023} methodology. Our model achieves $98\%$ accuracy
        on the \texttt{MNIST} dataset. The loss function $L(\theta) = \sum_{i=1}^{n} l_i$
        converges in $O(n\log n)$ time.
        """
        result = self.cleaner.remove_latex(input_text)
        
        # Check that LaTeX is removed but content is preserved
        self.assertNotIn("\\textbf", result)
        self.assertNotIn("\\emph", result)
        self.assertNotIn("\\cite", result)
        self.assertNotIn("\\texttt", result)
        self.assertIn("Abstract:", result)
        self.assertIn("machine learning", result)
        self.assertIn("98", result)
        self.assertIn("accuracy", result)
    
    def test_news_article_cleaning(self):
        """Test cleaning of news article HTML"""
        input_text = """
        <article class="news">
            <h1>Breaking: AI Model Achieves 99% Accuracy</h1>
            <p class="lead">Researchers at <strong>Tech Corp</strong> announced today...</p>
            <div class="metadata">Published: 2024-01-01</div>
            <!-- This is a comment -->
            <script>alert('test');</script>
        </article>
        """
        result = self.cleaner.clean_text(input_text)
        
        # Check that HTML is removed but content is preserved
        self.assertNotIn("<article", result)
        self.assertNotIn("<h1>", result)
        self.assertNotIn("<!--", result)
        self.assertNotIn("<script>", result)
        self.assertIn("Breaking:", result)
        self.assertIn("AI Model", result)
        self.assertIn("99", result)
        self.assertIn("Tech Corp", result)
    
    def test_combined_cleaning_pipeline(self):
        """Test full cleaning pipeline with both HTML and LaTeX"""
        input_text = r"""
        <p>According to \textbf{Smith et al.} the formula $E = mc^2$ shows that...</p>
        """
        # First remove LaTeX
        result = self.cleaner.remove_latex(input_text)
        # Then clean HTML
        result = self.cleaner.clean_text(result)
        
        self.assertNotIn("<p>", result)
        self.assertNotIn("\\textbf", result)
        self.assertNotIn("$", result)
        self.assertIn("Smith et al.", result)
        
        self.assertIn("E", result)
        self.assertIn("mc2", result)


if __name__ == '__main__':
    unittest.main()