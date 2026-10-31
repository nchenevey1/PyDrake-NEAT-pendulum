"""Export a GraphPolicy as a dependency-free C file (float32, fully unrolled).

  python -m pendulum.export runs/x/best_val.json policy.c

API:
  void  policy_reset(void);                        // clear recurrent state (episode start)
  float policy_observe_and_act(const float s[6]);  // s = [x, th1, th2, xd, w1, w2] -> action in [-1, 1]
  float policy_act(const float obs[9]);            // if you compute the 9-D observation yourself
"""

import sys

import numpy as np

from .policy import ACT_RELU, ACT_TANH, GraphPolicy
from .task import DEFAULT_TASK, TaskConfig, task_for_policy


def _f(x: float) -> str:
    r = repr(float(np.float32(x)))  # always has '.' or 'e'
    return f"{r}f"


def to_c(p: GraphPolicy, task: TaskConfig = None, name: str = "policy") -> str:
    """`task` defaults to the task with the policy's stored observation scales."""
    task = task if task is not None else task_for_policy(p)
    size = p.n_in + p.n_nodes
    rec = p.is_recurrent
    lines = [
        f"/* Auto-generated controller: {p.describe()} */",
        "#include <math.h>",
        "#include <string.h>",
        "",
        f"static float {name}_buf[{size}];",
        "",
        f"void {name}_reset(void) {{ memset({name}_buf, 0, sizeof({name}_buf)); }}",
        "",
        f"float {name}_act(const float obs[{p.n_in}]) {{",
        f"    float *b = {name}_buf;",
    ]
    if rec:
        lines += [f"    float prev[{size}];", f"    memcpy(prev, b, sizeof(prev));"]
    lines.append(f"    for (int i = 0; i < {p.n_in}; ++i) b[i] = obs[i];")
    for k in range(p.n_nodes):
        j = p.n_in + k
        terms = [_f(p.bias[k])]
        for s, w, d in zip(p.src[p.dst == j], p.w[p.dst == j], p.delayed[p.dst == j]):
            if w != 0:
                terms.append(f"{_f(w)} * {'prev' if d else 'b'}[{s}]")
        expr = " + ".join(terms)
        a = p.act[k]
        if a == ACT_TANH:
            expr = f"tanhf({expr})"
        elif a == ACT_RELU:
            expr = f"fmaxf(0.0f, {expr})"
        lines.append(f"    b[{j}] = {expr};")
    out = f"b[{p.out_pos}]"
    if p.action_mode == 1:
        out = f"tanhf({out})"
    lines += [f"    float a = {out};", "    return a > 1.0f ? 1.0f : (a < -1.0f ? -1.0f : a);", "}", ""]
    clip = "fminf(1.0f, fmaxf(-1.0f, {}))"
    lines += [
        f"float {name}_observe_and_act(const float s[6]) {{",
        "    float s1 = sinf(s[1]), c1 = cosf(s[1]), s2 = sinf(s[2]), c2 = cosf(s[2]);",
        "    float o[9] = {",
        f"        s[0] / {_f(task.limit_x)},",
        f"        {clip.format(f's[3] / {_f(task.max_v_cart)}')},",
        "        s1, c1,",
        f"        {clip.format(f's[4] / {_f(task.max_w_pole)}')},",
        "        s2, c2,",
        f"        {clip.format(f's[5] / {_f(task.max_w_pole)}')},",
        "        s1 * s2 + c1 * c2};",
        f"    return {name}_act(o);",
        "}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    pol = GraphPolicy.load(sys.argv[1])
    src = to_c(pol)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(src)
    else:
        print(src)
