from pollos_petrel import scatter_plot, plot_tarsus_vs_wing


def test_scatter_plot():
    fig, ax = scatter_plot("Longitud_ala", "Longitud_tarso", "target")
    expected_xlabel = "Longitud_ala"
    obtained_xlabel = ax.get_xlabel()
    assert expected_xlabel == obtained_xlabel
    expected_ylabel = "Longitud_tarso"
    obtained_ylabel = ax.get_ylabel()
    assert expected_ylabel == obtained_ylabel


def test_plot_tarsus_vs_wing():
    fig, ax = plot_tarsus_vs_wing()
    expected_xlabel = "Longitud_ala"
    obtained_xlabel = ax.get_xlabel()
    assert expected_xlabel == obtained_xlabel
    expected_ylabel = "Longitud_tarso"
    obtained_ylabel = ax.get_ylabel()
    assert expected_ylabel == obtained_ylabel
