"""Multi-run fit report and figures share one numeric run ordering.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Both the fit report (iterated from the results list, sorted by
run_sort_key) and the saved figures (listed in the Results tree, sorted
by fit_run_sort_key) must follow the same numeric run ordering: run
numbers sort numerically rather than lexically, embedded run tokens like
"run_10" are handled, non-numeric runs sort last, and the two sort keys
agree end to end.

Depends on: gui.analysis.project (run_sort_key), gui.results_tab
(fit_run_sort_key).
"""
from gui.analysis.project import run_sort_key
from gui.results_tab import fit_run_sort_key


def test_run_numbers_sort_numerically():
    runs = ["2", "10", "1", "11"]
    assert sorted(runs, key=run_sort_key) == ["1", "2", "10", "11"]


def test_run_numbers_with_embedded_token():
    runs = ["run_2", "run_10", "run_1"]
    assert sorted(runs, key=run_sort_key) == ["run_1", "run_2", "run_10"]


def test_non_numeric_runs_sort_last():
    runs = ["5", "merged", "3"]
    assert sorted(runs, key=run_sort_key) == ["3", "5", "merged"]


def test_figure_filenames_sort_by_run_number():
    files = ["fit_run_10.png", "fit_run_2.png",
             "fit_run_1.png", "fit_run_11.png"]
    assert sorted(files, key=fit_run_sort_key) == [
        "fit_run_1.png", "fit_run_2.png",
        "fit_run_10.png", "fit_run_11.png"]


def test_report_and_figures_share_one_ordering():
    run_numbers = ["10", "2", "1"]
    report_order = sorted(run_numbers, key=run_sort_key)
    fig_files = [f"fit_run_{r}.png" for r in run_numbers]
    figure_order = [f[len("fit_run_"):-len(".png")]
                    for f in sorted(fig_files, key=fit_run_sort_key)]
    assert report_order == figure_order == ["1", "2", "10"]
