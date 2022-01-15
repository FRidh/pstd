import sys
import pytest

try:
    from pstd.pstd_using_cuda import exp, add
except ImportError:
    pass


@pytest.mark.skipif("cupy" not in sys.modules, reason="requires cuda")
class TestCuda:
    """
    Cuda related tests.
    """

    def test_add(self):

        a = np.random.randn(100).astype("complex64") + 1j * np.random.randn(100).astype(
            "complex64"
        )
        b = np.random.randn(100).astype("complex64") + 1j * np.random.randn(100).astype(
            "complex64"
        )

        assert_array_almost_equal(a + b, add(a, b))

    def test_exp(self):

        theta = np.linspace(0.0, 2.0 * np.pi, 100).astype("complex64")
        assert_array_almost_equal(np.exp(1j * theta), exp(1j * theta))

        a = (np.random.randn(100) + 1j * np.random.randn(100)).astype("complex64")
        assert_array_almost_equal(np.log(np.exp(a)), np.log(exp(a)))

        x = np.arange(50).astype("complex64")
        assert_array_almost_equal(np.log(np.exp(x)), np.log(exp(x)))
