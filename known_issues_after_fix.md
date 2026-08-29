# Known issues after ToolRepo Direct grounding fix

Commit under review: `4efd666`

These observations came from the post-fix real ALFWorld smoke runs.  They are
not part of the Tool executable-grounding / attempt-evidence root cause fixed
by that commit, so they were deliberately not repaired in the same change.

## Benchmark win / Atomic contract mismatch for look tasks

- Run: `runs/smoke_toolrepo_direct_grounding_stats_v4`
- Task: `alfworld_24_look_at_obj_in_light`
- Observation: ALFWorld reported success, while the final Atomic validator did
  not certify its declared Effect (`benchmark_goal_contract_mismatch`).
- Safety behavior: the episode was marked `learning_eligible=false`, so it did
  not write success learning evidence.
- Follow-up family: Effect/witness semantics and Atomic validation (C/H), not
  Tool Direct grounding (B/E/F/K).

## Real warm Direct was not reached by the post-fix tiny runs

- Balanced 6x1 run: 3/6 successes, all six nodes Dynamic, Direct=0.
- Targeted five-task `pick_two_obj_and_place` run: 0/5 successes
  (`action_cycle` x2, `timeout_or_done` x3), so no Tool was mined and no warm
  Direct attempt could occur.
- The repaired Direct and Direct-failure accounting paths are covered by
  deterministic full-runtime regression tests, but the real-ALFWorld warm
  Direct smoke gate remains outstanding.

Until these smoke-health items are resolved or a clean real warm Direct smoke
is obtained, do not treat the current smoke results as authorization to start
the formal 120/60 experiment.
