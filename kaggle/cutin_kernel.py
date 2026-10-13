"""Kaggle script kernel (CPU): rerun the two cut-in sampling studies and write the CSVs to /kaggle/working."""
import glob
import os
import shutil
import subprocess
import sys

src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/cutin-atlas/src/cutin/study_strata.py", recursive=True)[0]))
root = os.path.dirname(src)
work = "/kaggle/temp/cutin-atlas"
shutil.copytree(root, work, dirs_exist_ok=True)
os.makedirs(f"{work}/results", exist_ok=True)
env = dict(os.environ, PYTHONPATH=f"{work}/src")
print("cpus:", os.cpu_count(), flush=True)
procs = [subprocess.Popen([sys.executable, "-m", f"cutin.{m}"], cwd=work, env=env,
                          stdout=open(f"/kaggle/working/{m}.log", "w"), stderr=subprocess.STDOUT)
         for m in ("study_strata", "study_cluster")]
codes = [p.wait() for p in procs]
for f in glob.glob(f"{work}/results/*.csv"):
    shutil.copy(f, "/kaggle/working/")
print("exit codes", codes, flush=True)
sys.exit(max(codes))
