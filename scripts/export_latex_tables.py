"""Export tables from v2 results; equivalence and inconclusive remain distinct."""

from spatial_graph_bench.analysis.reporting import generate_reports
from spatial_graph_bench.utils.paths import get_project_root


def generate_latex_table():
    generate_reports(get_project_root())


if __name__ == "__main__":
    generate_latex_table()
