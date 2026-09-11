# Esmakker Whist Rules Implemented

This document describes the Esmakker variant used by `esmakker_env.py`.

## Round order

1. A dealer is selected. The first round uses a random dealer.
2. Bidding starts with the player left of the dealer and proceeds clockwise.
3. A player may bid any contract higher than the current bid or pass. Passing is final for that round.
4. If all four players pass, the same deal automatically becomes Paskrig.
5. The highest bidder is declarer. Declarer names trump for numeric contracts, then names a partner suit different from trump.
6. Thirteen tricks are played. The winner of a trick leads the next trick.
7. The dealer moves clockwise after a numeric or Paskrig round. The same dealer deals again after a nolo contract.

## Contracts

| Bid | Type | Target |
|---|---|---|
| 7 to 13 | Partnership | Declarer team must win at least the named number of tricks. |
| Sol | Nolo | Declarer alone must win at most one trick. |
| Ren sol | Nolo | Declarer alone must win zero tricks. |
| Bordlægger | Nolo | Declarer alone must win zero tricks; declarer's hand is visible to opponents. |

For all nolo contracts, ace is the lowest card in normal tricks. This is the
same ace-low ranking used by Paskrig; numeric contracts remain ace-high except
for the documented special partner-card rule.

The bid order is `7, 8, Sol, 9, 10, Ren sol, 11, 12, Bordlægger, 13`.

## Paskrig

Paskrig is a four-player free-for-all with no declarer, partnership, partner
card or trump suit. Ace is low only in Paskrig, so the rank order is
`Ace, 2, 3, ..., Queen, King`. The highest card in the led suit wins the
trick, and that player leads the next trick. Each player tries to take as few
tricks as possible.

The player or players with the fewest tricks win. Each other player pays the
difference between their trick count and the winning trick count into one
combined pot. Tied winners split that pot equally; fractional kroner are
preserved rather than rounded, which keeps settlement exactly zero-sum.

Example: `[1, 4, 4, 4]` settles as `[+9, -3, -3, -3]`. A tie such as
`[1, 1, 5, 6]` settles as `[+4.5, +4.5, -4, -5]`.

`PASKRIG_TRIGGER` defaults to `auto_on_all_pass`. The alternate
`callable_bid` value enables direct calls to `game.bid(player, "paskrig")`.

### Engine assumptions

Normal tricks are ace-high; `_rank()` explicitly switches to ace-low for
Paskrig. Numeric settlement assumes a declarer partnership and nolo settlement
assumes one declarer against three opponents; `settle()` handles Paskrig first
as a separate free-for-all branch so neither assumption leaks into its payout.

## Partner card

For numeric contracts, declarer names a partner suit after naming trump. The suit must not be trump. The partner card is the ace of the named suit. The holder becomes declarer's partner, but the partner card is not announced.

When the partner suit is led, its holder must play the partner card if it remains in hand. This reveals the partnership.

If declarer holds all four aces, declarer instead names a non-trump suit whose king is the partner card. The ace in that suit is ranked below the two for that suit only; aces in the other suits remain highest.

## Settlement

For numeric contracts, the contract value on success is:

`base bid value + (team tricks - 6)`

Base values are: 7 = 0, 8 = 1, 9 = 2, 10 = 3, 11 = 4, 12 = 5, 13 = 25.

If a numeric contract fails, each declarer-team player pays one defender:

`base bid value + 2 * missing tricks`

Example: bidding 9 and taking 7 tricks costs `2 + (2 * 2) = 6` per declarer-team player.

Nolo contract values are fixed: Sol = 2, Ren sol = 4, Bordlægger = 8. On success, each opponent pays the declarer; on failure, declarer pays each opponent.
