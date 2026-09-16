"""Reproduce the frozen 10,000 draws and verify the group contributions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import generate_data

if __name__ == "__main__":
    generate_data.main()
    root = Path(__file__).resolve().parents[1]
    actual = pd.read_csv(root / "outputs/group_draws.csv")
    expected = pd.read_csv(root / "inputs/reference_final_group_draws.csv")
    expected = expected[actual.columns]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    result = {"reference_draws_match": True, "maximum_absolute_difference": float(np.max(np.abs(actual.to_numpy()-expected.to_numpy()))), "numpy": np.__version__, "pandas": pd.__version__}
    (root / "outputs/reproduction_check.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
