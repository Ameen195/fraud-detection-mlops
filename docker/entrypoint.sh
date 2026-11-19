#!/bin/sh
# The registry in mlflow.db stores absolute host paths (e.g. file:///D:/.../mlruns).
# Rewrite them to /app/mlruns in a private copy; the mounted original is never touched.
set -e
python - <<'PY'
import re, shutil, sqlite3
shutil.copy("/app/mlflow.db", "/tmp/mlflow.db")
con = sqlite3.connect("/tmp/mlflow.db")
con.create_function("fix", 1, lambda v: re.sub(r"file:(//)?/?[A-Za-z]:/.*?/mlruns", "file:///app/mlruns", v) if v else v)
for (t,) in con.execute("select name from sqlite_master where type='table'").fetchall():
    for _, c, ty, *_ in con.execute(f'pragma table_info("{t}")').fetchall():
        if "CHAR" in ty.upper() or "TEXT" in ty.upper():
            con.execute(f'update "{t}" set "{c}" = fix("{c}") where "{c}" like \'%:/%mlruns%\'')
con.commit()
PY
export MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db
exec "$@"
