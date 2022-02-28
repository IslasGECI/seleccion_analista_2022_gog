import pytest

from pollos_petrel import read_training_dataset, scatter_plot


def test_scatter_plot():
    fig, ax = scatter_plot("Longitud_ala", "Longitud_tarso", "target")
    assert ax.get_xlabel() == "Longitud_ala"
    assert ax.get_ylabel() == "Longitud_tarso"
