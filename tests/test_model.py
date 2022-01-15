import pytest
import numpy as np
from pstd import Model, PSTD, Medium


class TestModel:
    """
    Test Model.
    """

    def test_init_abstract_class(self):
        """
        __init__
        """
        f_max = 100.0
        with pytest.raises(RuntimeError):
            model = Model(f_max)


class TestMedium:
    """Test Medium."""

    def test_init_default(self):
        medium = Medium()

    def test_soundspeed_for_calculation(self):
        """Test when not used in a model."""

        # Single value. Can return
        medium = Medium()
        medium.soundspeed_for_calculation

        # Grid. Needs to know Model grid/pml.
        medium = Medium(np.ones((10, 10)))
        with pytest.raises(ValueError):
            medium.soundspeed_for_calculation
