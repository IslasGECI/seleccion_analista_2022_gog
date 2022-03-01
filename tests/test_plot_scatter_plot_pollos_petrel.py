from pollos_petrel import scatter_plot, plot_tarsus_vs_wing


def test_scatter_plot():
    fig, ax = scatter_plot("Longitud_ala", "Longitud_tarso", "target")
    assert ax.get_xlabel() == "Longitud_ala"
    assert ax.get_ylabel() == "Longitud_tarso"


def test_plot_tarsus_vs_wing():
    fig, ax = plot_tarsus_vs_wing()
    assert ax.get_xlabel() == "Longitud_ala"
    assert ax.get_ylabel() == "Longitud_tarso"
