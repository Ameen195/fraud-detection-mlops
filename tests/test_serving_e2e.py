"""/score must equal the offline model score on real validation rows. Needs local artifacts.

Set FRAUDGUARD_API_URL (e.g. http://localhost:8000) to check a running container instead of
the in-process app; the offline score always comes from the local registry model.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from fraudguard.data.load import load_config
from fraudguard.models.train import load_xy
from fraudguard.serving.app import create_app, row_to_payload

cfg = load_config()
FULL = Path(cfg["paths"]["processed_dir"]) / "full.parquet"
pytestmark = pytest.mark.skipif(
    not (FULL.exists() and Path("mlflow.db").exists()), reason="needs data + MLflow registry"
)


def test_score_equals_offline_score():
    threshold = json.loads(Path(cfg["paths"]["results_dir"], "threshold.json").read_text())[
        "threshold"
    ]
    url = os.environ.get("FRAUDGUARD_API_URL", "")
    with TestClient(create_app()) as c:
        s = c.app.state.s
        api = requests.Session() if url else c
        assert api.get(f"{url}/model").json()["version"] == s.info["version"]
        X, _, split = load_xy(cfg)
        rows = X[(split == "valid").to_numpy()].sample(5, random_state=0)
        raw = pd.read_parquet(FULL).loc[rows.index]
        offline = s.model.predict_proba(rows[s.names])[:, 1]
        for (i, row), p_off in zip(raw.iterrows(), offline, strict=True):
            resp = api.post(f"{url}/score", json=row_to_payload(row))
            assert resp.status_code == 200, resp.text[:500]
            r = resp.json()
            assert r["probability"] == pytest.approx(p_off, abs=1e-9), i
            assert r["decision"] == ("flag" if p_off >= threshold else "allow")
