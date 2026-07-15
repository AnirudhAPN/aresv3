"""Tests for utils/text.py — strips model-embedded step numbers without
corrupting legitimate content that happens to start with a digit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.text import strip_leading_ordinal


def test_strips_common_ordinal_prefixes():
    assert strip_leading_ordinal("1. Collect data") == "Collect data"
    assert strip_leading_ordinal("2) Train the model") == "Train the model"
    assert strip_leading_ordinal("Step 3: Evaluate results") == "Evaluate results"
    assert strip_leading_ordinal("4 - Deploy to staging") == "Deploy to staging"
    assert strip_leading_ordinal("  5.   Extra whitespace") == "Extra whitespace"


def test_leaves_unnumbered_text_untouched():
    assert strip_leading_ordinal("Collect data from the whitelisted source") == "Collect data from the whitelisted source"


def test_does_not_strip_legitimate_leading_numbers():
    # These start with a digit but are not step markers — no space follows
    # the punctuation, so they must survive untouched.
    assert strip_leading_ordinal("5-fold cross-validation setup") == "5-fold cross-validation setup"
    assert strip_leading_ordinal("24-hour monitoring window") == "24-hour monitoring window"
    assert strip_leading_ordinal("3.14 rounding constant applied to metrics") == "3.14 rounding constant applied to metrics"


def test_only_strips_once_at_the_start():
    # A number appearing later in the sentence must not be touched.
    assert strip_leading_ordinal("1. Run 2 training epochs") == "Run 2 training epochs"


if __name__ == "__main__":
    test_strips_common_ordinal_prefixes()
    test_leaves_unnumbered_text_untouched()
    test_does_not_strip_legitimate_leading_numbers()
    test_only_strips_once_at_the_start()
    print("All text util tests passed.")
