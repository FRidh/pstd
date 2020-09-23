{ buildPythonPackage
, acoustics
, h5py
, numpy
, scipy
, matplotlib
, numexpr
, pyyaml
}:

buildPythonPackage rec {
  pname = "pstd";
  version = "dev";

  src = ./.;

  propagatedBuildInputs = [
    acoustics
    h5py
    numpy
    scipy
    matplotlib
    numexpr
    pyyaml
  ];

  # No tests
  doCheck = false;

}