"""Hypothesis test: a small MLP with a clock input (obs + t/T_nom, saturating) can imitate the
time-indexed expert and solve the task. Evaluated with the exact task rules (python loop)."""
import sys, time, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from distill import expert_rollout, make_net, ClassicController
from pendulum.classic import rollout_controller
from pendulum.evaluation import summarize, val_ics, test_ics
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS, sample_initial_states
torch.set_num_threads(1)
H = int(sys.argv[1]); USE_CLOCK = sys.argv[2] == 'clock'; noise = 0.15
expert = ClassicController.load('results/classic/controller.npz')
T_nom = len(expert.u_nom)                      # steps of the time-varying part
def clock(k): return min(k, T_nom) / T_nom if USE_CLOCK else 0.0
rng = np.random.default_rng(0); torch.manual_seed(0)
X, Y, PH = [], [], []
for ic in sample_initial_states(256, rng, DEFAULT_TASK.ic_noise):
    o, a = expert_rollout(expert, ic, noise, rng, return_labels=True)
    X.append(np.hstack([o, np.array([[clock(k)] for k in range(len(o))])])); Y.append(a)
    PH.append(np.arange(len(o)) < T_nom + 50)
X = torch.as_tensor(np.vstack(X)); Y = torch.as_tensor(np.concatenate(Y))[:, None]
PH = np.concatenate(PH)
w = np.where(PH, 0.5 / PH.sum(), 0.5 / (~PH).sum())          # equal total weight per phase
sampler = torch.distributions.Categorical(torch.as_tensor(w))
layers = [torch.nn.Linear(10, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(), torch.nn.Linear(H, 1), torch.nn.Tanh()]
net = torch.nn.Sequential(*layers).double()
opt = torch.optim.Adam(net.parameters(), 3e-3); steps = 20000
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
t0 = time.time()
for i in range(steps):
    idx = sampler.sample((2048,))
    loss = torch.mean((net(X[idx]) - Y[idx]) ** 2); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
n_params = sum(p.numel() for p in net.parameters())
class Student:
    def reset(self): self.k = 0
    def __call__(self, o):
        with torch.no_grad():
            a = float(net(torch.as_tensor(np.append(o, clock(self.k)))[None])[0, 0])
        self.k += 1; return a
r = summarize(rollout_controller(Student(), val_ics(64), TARGET_PHYSICS), DEFAULT_TASK)
print(f"{'clock' if USE_CLOCK else 'reactive'}-MLP 10-{H}-{H}-1 ({n_params} params, {len(X)} samples, train {time.time()-t0:.0f}s): "
      f"mse {float(loss):.2e}  VAL success {r['success_rate']:.2f} streak {r['mean_streak_s']:.1f}s", flush=True)
if r['success_rate'] > 0.5:
    t = summarize(rollout_controller(Student(), test_ics(256), TARGET_PHYSICS), DEFAULT_TASK)
    print(f"  TEST success {t['success_rate']:.3f}", flush=True)
torch.save(net, f'results/experiments/{sys.argv[2]}_mlp{H}.pt')
