# Curriculum and advancement design

## Short answers

| Question | Answer | Why |
|---|---|---|
| Change the 2.5 s swing-up time? | **No.** Keep it as the task definition. | The classic controller swings up in 2.0 s with 85 N, so it is feasible. Changing it changes the problem. |
| Use the allowed swing-up time as the main curriculum knob? | **Not on its own.** | From rest with a 10 s deadline the first level is very hard to enter: the policy must find swing-up *and* balance at once. The 45-param MLP never entered it in 2B steps; the 89-param MLP needed ~2,800 generations (1.2B steps). |
| Is it useful at all? | **Yes, for the final timing squeeze.** | Once the 89-param MLP entered, the deadline fell from 10 s to 2.6 s in ~800 generations. It stalled 0.1 s short of the real task. |
| How to combine them? | **Start-state curriculum with deadline slack**, then shrink the slack to 0. | The start-state curriculum enters easily (its first level is balance only) but slows near the start of the swing-up. The deadline curriculum enters badly but compresses timing quickly. |
| Keep the physics curriculum? | **Only as a baseline.** | It never produced a policy that balances, even at its own training physics (see below). |
| What should change? | **Where episodes start (start-state curriculum) and how advancement is decided (mastery rule).** | First and only neural success on the real task came from this. |

## Why the physics curriculum failed

1. **It advanced without mastery.** It moved on when the single best sampled candidate was
   upright 20 % of the episode. "Swing up, wobble 4 s, fall" passes this, so difficulty rose
   before any policy could balance.
2. **The easy end is a different problem.** At g = 1, drag 3, 1000 N the dynamics are dominated
   by actuation and damping. Skills learned there do not carry over; champions failed a few
   curriculum steps back.
3. **It never went back.** Once a step was too large, the population stalled for good (stalls
   at difficulty 0.18–0.91 out of 1.0).

## Recommended curriculum: start states (S2)

The physics is always the real one. Only the start of the episode moves.

* A **reference swing-up** gives a sequence of states `S_ref[k]`. It currently comes from the
  classic controller's optimized trajectory, upright at 2.1 s.
* **Level k** starts at `S_ref[k]` plus the task's ±0.05 noise, with:
  * remaining deadline = 2.5 s − k·dt;
  * episode length = 20 s − k·dt, so the balance horizon is unchanged.
* Training starts near upright (k = 210, balance only) and moves k back toward 0. **k = 0 is
  exactly the real task**: hanging start, 2.5 s deadline, 20 s episode.
* The score is the same sparse streak score.

This is the reverse-curriculum idea of Florensa et al. (CoRL 2017) and of Salimans & Chen
(2018, single demonstration).

## Advancement rule (all curricula)

| | Old (physics curriculum) | New (mastery rule) |
|---|---|---|
| Policy measured | best sampled candidate | the policy we would deploy (CMA-ES mean, NEAT champion) |
| Test starts | the generation's 5 training starts | 16 fresh starts |
| Pass condition | upright ≥ 20 % of the episode | full-episode success ≥ 80 % |
| How often | every generation | every 5 generations |
| Step size | 0.005 difficulty | 0.05 s of reference time (5 control steps) |
| Forgetting | not handled | 50 % of training starts drawn from a random mastered level |
| Stuck | stays forever | back off one level after 300 generations with no full success |
| Physics | changes (domain shift) | always the real physics |

Values were set once from first principles and are **not tuned**:
* 80 % on 16 starts makes a lucky pass very unlikely;
* replay of 50 % is the usual choice;
* 300 generations is ~1 % of a typical run.

### Evidence for the new rule

* With the old-style rule (best candidate, single pass) the start-state curriculum advanced
  quickly, then **collapsed to zero** and could not recover.
* With the mastery rule, every level was mastered at 100 %:
  * 45-param MLP, seed 0: from 2.1 s down to 0.45 s of reference time in 2B steps;
  * 45-param MLP, seed 1: to 0.60 s, and **9.0 % test success** on the real task;
  * 61-param RNN: to 1.30 s, no real-task success within 2B steps.

## Refinements to test next (in this order)

1. **Deadline slack (combined curriculum).** At every level give the remaining time plus a
   slack (e.g. +1 s). After reaching k = 0, shrink the slack to 0 with the same mastery rule
   (step 0.1 s). This uses each knob where it worked.
2. **Adaptive step.** Progress slows in the early, energy-pumping part of the swing-up. Halve the
   start-time step after each back-off (minimum 0.01 s).
3. **Final phase at the real task.** Keep 25 % replay of earlier levels until the budget ends;
   select models on a larger validation set (256 starts).
4. **No reference (S3).** Build start states by random walks backward from upright, as in
   Florensa et al. This shows whether the classic trajectory is needed at all.
5. **Several references.** Use reference swing-ups in both directions and from several
   optimizer seeds, so the network does not just learn one demonstration.

## What stays fixed

Task constants, physics, the sparse score, held-out validation and test starts, and the env-step
budget are identical across curricula, so differences in results come from the curriculum alone.
