"""Test cases for the utils module."""

import pytest

from src.neuronal_activity.utils import generate_codename


@pytest.mark.parametrize("_", range(100))
@pytest.mark.usefixtures("_")
def test_generate_codename_content() -> None:
    """Test the content of generated codenames.

    Verifies that the generated codename consists of an adjective and a noun
    from predefined lists. This test is parameterized to run multiple times.

    Args:
        _: Unused parameter for parameterized testing.

    Raises:
        AssertionError: If the generated codename does not match the expected format
            or if the adjective or noun is not from the predefined lists.

    """
    codename = generate_codename()
    adjective, noun = codename.split("-")

    # Test that the adjective is from the predefined list
    assert adjective in [
        "adventurous",
        "bold",
        "brave",
        "bright",
        "calm",
        "charming",
        "cheerful",
        "daring",
        "delightful",
        "eager",
        "elegant",
        "energetic",
        "fancy",
        "faithful",
        "fearless",
        "generous",
        "gentle",
        "graceful",
        "happy",
        "honest",
        "humble",
        "inventive",
        "jolly",
        "joyful",
        "keen",
        "kind",
        "lively",
        "loyal",
        "loving",
        "merry",
        "modest",
        "motivated",
        "nice",
        "noble",
        "optimistic",
        "outgoing",
        "patient",
        "polite",
        "proud",
        "quick",
        "reliable",
        "respectful",
        "silly",
        "sincere",
        "spirited",
        "thoughtful",
        "trustworthy",
        "upbeat",
        "vibrant",
        "warm",
        "witty",
        "youthful",
        "zany",
        "zealous",
    ]

    # Test that the noun is from the predefined list
    assert noun in [
        "alligator",
        "anteater",
        "armadillo",
        "badger",
        "bat",
        "bear",
        "beaver",
        "buffalo",
        "camel",
        "cheetah",
        "chimpanzee",
        "chipmunk",
        "chinchilla",
        "crocodile",
        "dolphin",
        "eagle",
        "elephant",
        "falcon",
        "ferret",
        "flamingo",
        "fox",
        "gerbil",
        "giraffe",
        "gorilla",
        "guinea pig",
        "hamster",
        "hedgehog",
        "hippopotamus",
        "kangaroo",
        "koala",
        "lemur",
        "leopard",
        "lion",
        "lynx",
        "meerkat",
        "mole",
        "monkey",
        "mouse",
        "orangutan",
        "otter",
        "owl",
        "panda",
        "parrot",
        "peacock",
        "penguin",
        "porcupine",
        "rabbit",
        "raccoon",
        "rat",
        "rhinoceros",
        "shark",
        "skunk",
        "sloth",
        "sparrow",
        "squirrel",
        "swan",
        "tiger",
        "weasel",
        "whale",
        "wolf",
        "woodpecker",
        "zebra",
    ]
