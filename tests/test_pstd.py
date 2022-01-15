import pytest
from pstd import PSTD, PML
import collections

model_parameters = [
    {
        "maximum_frequency": 1000.0,
        "cfl": 0.05,
        "size": (15, 15),
    },
    {
        "maximum_frequency": 1000.0,
        "cfl": 0.10,
        "size": (20, 20),
    },
]


class TestPSTD:
    """
    Test Model.
    """

    # @pytest.fixture(params=[0.05, 0.10])
    # def cfl(self, request):
    # return request.param

    @pytest.fixture(params=model_parameters)
    def model(self, request):
        kwargs = request.param
        kwargs["pml"] = PML(depth=10.0)
        model = PSTD(**kwargs)
        return model

    @pytest.fixture(params=[10])
    def steps(self, request):
        return request.param

    @pytest.fixture(params=[0.001])
    def seconds(self, request):
        return request.param

    def test_init(self, model):
        """Initialization of model.

        If this passes, then the models returned by the fixture
        were initialized successfully.
        """
        pass

    def test_init_no_size(self):
        """
        __init__
        """
        f_max = 100.0
        with pytest.raises(RuntimeError):
            model = PSTD(maximum_frequency=f_max)

    def test_overview(self, model):
        model.overview()

    def test_properties(self, model):

        properties = ["cfl", "constant_field", "timestep"]

        for prop in properties:
            getattr(model, prop)

    def test_run_steps(self, model, steps):
        model.run(steps=steps)

    def test_run_seconds(self, model, seconds):
        model.run(seconds=seconds)


class TestPML:
    """Test PML.

    Should use Mock!
    """

    def setup_method(self, method):
        time = 1.0
        f_max = 100.0
        size = (100.0, 50.0)
        self.model = PSTD(maximum_frequency=f_max, size=size)

    def test_init_default(self):
        pass

    def test_init_custom(self):
        absorption_coefficient = (100.0, 100.0)
        depth = 5.0
        pml = PML(absorption_coefficient, depth)

    def test_depth(self):

        assert self.model.pml.depth == 0.0

        self.model.pml.depth_target = 1.0
        assert self.model.pml.depth != 0.0

    def test_nodes(self):

        assert self.model.pml.nodes == 0

        self.model.pml.depth_target = 1.0
        assert self.model.pml.nodes != 0.0

    def is_used(self):

        assert self.model.is_used

        self.model.settings["pml"]["use"] = False
        assert not self.model.is_used

    def generate_grid(self):

        with pytest.raises(ValueError):  # Zero size, cannot generate a PML.
            self.model.pml.generate_grid()

        self.model.pml.depth_target = 1.0
        pml = self.model.pml.generate_grid()

        assert isinstance(pml, collections.MutableMapping)
        assert isinstance(pml, dict)


# class TestCalculation():


# def test_
