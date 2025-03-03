"""
test_plotting.py - Test suite for plotting scripts
"""

import pytest
import subprocess
import re


def test_comp_plot_script(tmp_path):

    # Run the script
    try:
        result = subprocess.run(
            [
                "python",
                "run_comp_plots.py",
                "--mc",
                "./assets/evts_000_100.root",
                "--weights",
                "./assets/wgts.npz",
                "--pd",
                "./assets/truth_evts_000_100.root",
                "--store",
                tmp_path,
                "--use_test",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Script failed with return code {e.returncode} and output: {e.output}"
        )

    # Check that the plots were made
    assert (tmp_path / "pT_ll.png").exists()

    # Check that we got the correct W1 values
    output = result.stdout
    lines = output.splitlines()
    for line in reversed(lines):
        if re.search("Original Wasserstein One:", line):
            ogw1 = float(line.split()[-1])
        if re.search("Re-weighted Wasserstein One:", line):
            rw1 = float(line.split()[-1])

    assert ogw1 == pytest.approx(100.826, rel=1e-1)
    assert rw1 == pytest.approx(451.478, rel=1e-1)


def test_test_plot_script(tmp_path):

    # Run the script
    try:
        result = subprocess.run(
            [
                "python",
                "run_test_plots.py",
                "--f1",
                "./assets/evts_000_100.root",
                "--f2",
                "./assets/evts_200_300.root",
                "--name1",
                "MC",
                "--name2",
                "Pseudodata",
                "--step",
                "1",
                "--end_weights",
                "./assets/wgts.npz",
                "--store",
                tmp_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Script failed with return code {e.returncode} and output: {e.output}"
        )

    # Check that the plots were made
    assert (tmp_path / "pT_ll.png").exists()
