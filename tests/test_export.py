"""The exported C controller must match the Python reference (float32 tolerance)."""
import shutil
import subprocess

import numpy as np
import pytest

from pendulum.export import to_c
from pendulum.policy import build_arch
from pendulum.task import observe
from tests.test_policy import random_graph

CC = shutil.which("cc") or shutil.which("gcc")


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_export_matches_python(tmp_path):
    rng = np.random.default_rng(0)
    pols = [build_arch("mlp:4", 9, rng), build_arch("rnn:3", 9, rng), random_graph(rng)]
    states = rng.uniform(-1, 1, (30, 6)) * np.array([2, 3, 3, 5, 10, 10])
    for i, p in enumerate(pols):
        main = "#include <stdio.h>\n" + to_c(p) + "\nint main(void){ float s[6]; policy_reset();\n" \
               " while (scanf(\"%f %f %f %f %f %f\", s, s+1, s+2, s+3, s+4, s+5) == 6)" \
               " printf(\"%.7f\\n\", policy_observe_and_act(s)); return 0; }\n"
        (tmp_path / f"p{i}.c").write_text(main)
        exe = tmp_path / f"p{i}"
        subprocess.run([CC, "-O2", "-o", str(exe), str(tmp_path / f"p{i}.c"), "-lm"], check=True)
        inp = "\n".join(" ".join(f"{v:.7f}" for v in s) for s in states.astype(np.float32))
        out = subprocess.run([str(exe)], input=inp, capture_output=True, text=True, check=True).stdout
        got = np.array([float(x) for x in out.split()])
        step = p.make_stepper()
        ref = np.clip([step(observe(s.astype(np.float32).astype(np.float64))) for s in states], -1, 1)
        np.testing.assert_allclose(got, ref, atol=2e-4)
