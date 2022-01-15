{ buildPythonPackage

, flit-core

, acoustics
, h5py
, numba
, numpy
, scipy
, matplotlib
, numexpr
, pyyaml

, black
, nbconvert
, pylint
, pytest
, pytestCheckHook
}:

buildPythonPackage rec {
  pname = "pstd";
  version = "dev";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = [
    flit-core
  ];

  propagatedBuildInputs = [
    acoustics
    h5py
    numba
    numpy
    scipy
    matplotlib
    numexpr
    pyyaml
  ];

  checkInputs = [
    black
    nbconvert
    pylint
    pytest
    pytestCheckHook
  ];

  preCheck = ''
    echo "Checking formatting with black..."
    black --check pstd tests docs/conf.py
    echo "Static analysis with pylint..."
    pylint -E pstd
  '';

  passthru.allInputs = nativeBuildInputs ++ propagatedBuildInputs ++ checkInputs;

}