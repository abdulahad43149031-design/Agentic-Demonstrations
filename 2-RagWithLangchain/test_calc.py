from calc import square
import pytest

@pytest.mark.parametrize(
    "x, expected",
    [
        (2, 4),
        (3, 9),
        (4, 16),
    ]
)
def test_square(x, expected):
    assert square(x) == expected