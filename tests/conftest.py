import random

import pytest


@pytest.fixture
def seed():
    return random.getrandbits(32)
