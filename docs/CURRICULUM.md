# Curriculum and advancement design

## Short answers

| Question | Answer | Why |
|---|---|---|
| Change the 2.5 s swing-up time? | **No.** Keep it as the task definition. | The classic controller swings up in 2.0 s with 85 N, so it is feasible. Changing it changes the problem. |
| Use the allowed swing-up time as the main curriculum knob? | **No.** | From rest with a 10 s deadline, CMA-ES found no full success in 3,000+ generations (45 and 89 params). A longer deadline does not create intermediate successes: the policy still has to discover swing-up *and* balance at once. |
| Use it at all? | **Only as part of the start-state curriculum**, where the deadline is the time remaining on the reference (2.5 s − start time). An optional extra slack is listed under refinements. | Keeps every level a scaled-down copy of the real task. |
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
  * 45-param MLP, seed 1: the same pace;
  * 61-param RNN: advancing.

## Refinements to test next (in this order)

1. **Adaptive step.** Progress slows in the early, energy-pumping part of the swing-up (last
   ~0.5 s of the curriculum). Halve the step after each back-off (minimum 0.01 s).
2. **Final phase at k = 0.** Once the real task is reached, keep 25 % replay of earlier levels
   until the budget ends; select models only on real-task validation.
3. **No reference (S3).** Build start states by random walks backward from upright, as in
   Florensa et al. This shows whether the classic trajectory is needed at all.
4. **Several references.** Use reference swing-ups in both directions and from several
   optimizer seeds, so the network does not just learn one demonstration.
5. **Deadline slack (optional).** At each level allow +0.3 s extra, annealed to 0 before the next
   step back. Use only if (1) does not remove the stall near the start.

## What stays fixed

Task constants, physics, the sparse score, held-out validation and test starts, and the env-step
budget are identical across curricula, so differences in results come from the curriculum alone.
