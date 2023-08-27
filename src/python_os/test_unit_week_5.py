import src
import unittest

class TestRearrange(unittest.TestCase):
    def test_basic(self):
        testcase = "leon, eleazar"
        expected = "eleazar leon"
        self.assertEqual(src.python_os.week_5.rearrange_name(testcase),expected)

    def test_empty(self):
        testcase = ""
        excepted = ""
        self.assertEqual(src.python_os.week_5.rearrange_name(testcase), excepted)

    def test_double_name(self):
        textcase = "kennedy, John.f"
        expected = "John.f kennedy"
        self.assertEqual(src.python_os.week_5.rearrange_name(textcase),expected)

    def test_one_name(self):
        textcase = "leon"
        expected = "leon"
        self.assertEqual(src.python_os.week_5.rearrange_name(textcase), expected)

unittest.main()