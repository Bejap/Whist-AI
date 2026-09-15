# Phase 7: Full-game integration

## Goal

Compose the specialists into one legal controller without changing the rules engine.

## Controller behavior

At each decision, inspect the game phase and route to exactly one specialist:

```text
bidding -> bidding specialist
choose_trump / choose_partner -> declaration specialist
play -> card-play specialist
```

The controller owns phase transitions, masks, seat routing, fallback behavior, logging, and checkpoint compatibility checks. A fallback policy may be used for unavailable components, but it must be visible in evaluation reports.

## Compatibility

Every checkpoint records observation version, action-space version, rules version, policy role, device, and training configuration. Loading an incompatible specialist must fail clearly rather than silently adapting dimensions.

## Later joint fine-tuning

Joint fine-tuning is optional and comes only after the composed system is stable. Freeze one specialist at a time during ablations so any change in strength can be attributed.

## Exit criteria

- Complete games route decisions to the correct specialist.
- Mixed policies remain legal and deterministic under fixed seeds.
- Component failures and fallbacks are logged.
- Integrated performance is compared with the individual specialist baselines.
