import pathlib
import pytest

import nbformat
import nbconvert.preprocessors


NOTEBOOKS = list((pathlib.Path(__file__).parent.parent / "examples").glob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook(notebook):

    executor = nbconvert.preprocessors.ExecutePreprocessor(timeout=60)

    with open(notebook) as fin:
        contents = nbformat.read(fin, as_version=4)
        executor.preprocess(contents)
