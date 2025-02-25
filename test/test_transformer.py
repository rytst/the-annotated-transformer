import unittest
import transformer as t

def example_mask() -> None:
    print("hello")


class TestSubsequentMask(unittest.TestCase):
    def test_subsequent_mask(self) -> None:
        example_mask()


if __name__ == "__main__":
    unittest.main()
