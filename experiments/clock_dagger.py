"""DAgger with a clock input (labels are then a consistent function of (obs, t)), phase-balanced."""
import sys, time, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from distill import expert_rollout, ClassicController
from pendulum.classic import rollout_controller
from pendulum.rollout import physics_step
from pendulum.evaluation import summarize, val_ics, test_ics
from pendulum.task import DEFAULT_TASK as T, TARGET_PHYSICS as P, sample_initial_states, observe
torch.set_num_threads(1)
H = int(sys.argv[1]); ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ex = ClassicController.load('results/classic/controller.npz')
Tn = len(ex.u_nom); clock = lambda k: min(k, Tn) / Tn
rng = np.random.default_rng(1); torch.manual_seed(1)
net = torch.nn.Sequential(torch.nn.Linear(10, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(),
                          torch.nn.Linear(H, 1), torch.nn.Tanh()).double()
class Student:
    def reset(self): self.k = 0
    def __call__(self, o):
        with torch.no_grad(): a = float(net(torch.as_tensor(np.append(o, clock(self.k)))[None])[0, 0])
        self.k += 1; return a
def student_rollout(ic):
    """Student drives; expert labels each visited (obs, k). Stops at task termination."""
    st = Student(); st.reset(); ex.reset(); s = ic.copy(); X, Y, PH = [], [], []
    for k in range(T.max_steps):
        o = observe(s); X.append(np.append(o, clock(k))); Y.append(ex(o)); PH.append(k < Tn + 50)
        physics_step(s, st(o) * P.max_force, P.g, P.drag, T.dt)
        up = abs((s[1] - np.pi + np.pi) % (2*np.pi) - np.pi) < T.upright_tol and abs((s[2] + np.pi) % (2*np.pi) - np.pi) < T.upright_tol
        if not np.all(np.isfinite(s)) or abs(s[0]) > T.limit_x or (k + 1 > T.swing_up_steps and not up): break
    return X, Y, PH
X, Y, PH = [], [], []
for ic in sample_initial_states(64, rng, T.ic_noise):            # round 0: expert + DART noise
    o, a = expert_rollout(ex, ic, 0.1, rng, return_labels=True)
    X += [np.append(oi, clock(k)) for k, oi in enumerate(o)]; Y += list(a); PH += [k < Tn + 50 for k in range(len(o))]
t0 = time.time()
for it in range(ITERS):
    Xt = torch.as_tensor(np.array(X)); Yt = torch.as_tensor(np.array(Y))[:, None]; ph = np.array(PH)
    w = np.where(ph, 0.5 / ph.sum(), 0.5 / max((~ph).sum(), 1))
    sampler = torch.distributions.Categorical(torch.as_tensor(w))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for i in range(6000):
        idx = sampler.sample((2048,)); loss = torch.mean((net(Xt[idx]) - Yt[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    r = summarize(rollout_controller(Student(), val_ics(64), P), T)
    print(f"clock-DAgger H={H} it {it}: data {len(X)} wmse {float(loss):.2e} VAL success {r['success_rate']:.2f} "
          f"streak {r['mean_streak_s']:.1f}s swing-up reached {r['swing_up_s']} ({time.time()-t0:.0f}s)", flush=True)
    if r['success_rate'] >= 0.95: break
    for ic in sample_initial_states(32, rng, T.ic_noise):
        x, y, p = student_rollout(ic); X += x; Y += y; PH += p
n_params = sum(p.numel() for p in net.parameters())
t = summarize(rollout_controller(Student(), test_ics(256), P), T)
print(f"FINAL clock-DAgger H={H} ({n_params} params): TEST success {t['success_rate']:.3f}", flush=True)
torch.save(net, f'results/experiments/clockdagger_{H}.pt')
