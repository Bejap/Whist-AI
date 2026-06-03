"""Play or watch a Whist game with a trained AI agent.

Modes
-----
watch   – All four seats are controlled by the AI (original behaviour).
random  – One AI (P1) faces three random-but-valid opponents (P2/P3/P4).
play    – You play as P1; the remaining three seats are controlled by the AI.

Usage
-----
    python play.py                    # interactive mode menu
    python play.py --mode watch
    python play.py --mode random
    python play.py --mode play
"""

import sys
import copy

import numpy as np
import torch

from whist_env import WhistEnv, card_name, trump_name, SUITS, TEAMS, NUM_CARDS, NUM_PLAYERS

# ---------------------------------------------------------------------------
# Card display helpers
# ---------------------------------------------------------------------------

RANK_SHORT = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUIT_SHORT = ["C", "D", "H", "S"]


def card_short(card_id: int) -> str:
    """Return a short card label, e.g. 'QC', '10D', '6H'."""
    return f"{RANK_SHORT[card_id % 13]}{SUIT_SHORT[card_id // 13]}"


def print_trick_summary(trick_num: int, trick_cards: list) -> None:
    """Print a formatted trick summary.

    Example output (starting from the lead player):
        Trick 3.
        P3: 4C
        P4: QC
        P1: 6D
        P2: 8C
    """
    print(f"\n  Trick {trick_num}.")
    for player, card in trick_cards:
        print(f"  P{player + 1}: {card_short(card)}")


# ---------------------------------------------------------------------------
# Model / action helpers
# ---------------------------------------------------------------------------


def load_model():
    """Load the latest checkpoint."""
    from train import latest_checkpoint
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    ckpt_path, episode = latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found. Train the agent first with: python train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path} (episode {episode})")
    try:
        model = RecurrentPPO.load(ckpt_path, device="cpu")
        print("Model type: RecurrentPPO")
    except Exception:
        model = PPO.load(ckpt_path, device="cpu")
        print("Model type: PPO")
    return model


def _fallback_priors(model, obs, mask, samples: int = 32) -> np.ndarray:
    """Estimate masked action priors by repeated stochastic model sampling."""
    valid = np.where(mask > 0)[0]
    priors = np.zeros(NUM_CARDS, dtype=np.float32)
    if len(valid) == 0:
        return priors

    for _ in range(samples):
        action, _ = model.predict(obs, deterministic=False)
        action = int(action)
        if mask[action] <= 0:
            action = int(np.random.choice(valid))
        priors[action] += 1.0

    if priors.sum() <= 0:
        priors[valid] = 1.0 / len(valid)
    else:
        priors /= priors.sum()
    return priors


def policy_priors_and_value(model, obs, mask):
    """Return (priors, value) for one state.

    priors: np.ndarray shape (NUM_CARDS,), masked and normalised action probs.
    value: scalar value-head estimate for the same observation.
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    priors = None

    try:
        with torch.no_grad():
            logits = model.policy.get_distribution(obs_t).distribution.logits
        logits = logits.squeeze(0).cpu().numpy()
        logits[mask == 0] = -1e8
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs[mask == 0] = 0.0
        if probs.sum() <= 0:
            priors = _fallback_priors(model, obs, mask)
        else:
            priors = probs / probs.sum()
    except Exception:
        priors = _fallback_priors(model, obs, mask)

    try:
        with torch.no_grad():
            value = float(model.policy.predict_values(obs_t).cpu().numpy().reshape(-1)[0])
    except Exception:
        value = 0.0

    return priors, value


def _rollout_value(env, model, root_player: int, max_steps: int = 52) -> float:
    """Mutate env with a rollout and return normalized root-team score."""
    for _ in range(max_steps):
        if env.done:
            break
        obs = env._get_obs()
        mask = env.action_mask()
        priors, _ = policy_priors_and_value(model, obs, mask)
        valid = np.where(mask > 0)[0]
        if len(valid) == 0:
            break
        valid_probs = priors[valid]
        if valid_probs.sum() <= 0:
            valid_probs = np.ones(len(valid), dtype=np.float32) / len(valid)
        else:
            valid_probs = valid_probs / valid_probs.sum()
        action = int(np.random.choice(valid, p=valid_probs))
        env.step(action)

    team = TEAMS[root_player]
    team_diff = env.team_tricks[team] - env.team_tricks[1 - team]
    return team_diff / 13.0


def mcts_action(env, model, sims: int = 64, c_puct: float = 1.25) -> int:
    """Pick an action with a lightweight PUCT search at the root."""
    root_player = env.current_player
    obs = env._get_obs()
    mask = env.action_mask()
    valid = np.where(mask > 0)[0]
    if len(valid) == 1:
        return int(valid[0])

    priors, root_value = policy_priors_and_value(model, obs, mask)
    priors[mask == 0] = 0.0
    if priors.sum() <= 0:
        priors[valid] = 1.0 / len(valid)
    else:
        priors = priors / priors.sum()

    q = np.zeros(NUM_CARDS, dtype=np.float32)
    n = np.zeros(NUM_CARDS, dtype=np.float32)

    if sims <= 0:
        return int(valid[np.argmax(priors[valid])])

    for _ in range(sims):
        total_n = np.sum(n[valid]) + 1.0
        u = c_puct * priors * np.sqrt(total_n) / (1.0 + n)
        scores = q + u
        scores[mask == 0] = -1e9
        action = int(np.argmax(scores))

        env_sim = copy.deepcopy(env)
        env_sim.step(action)
        rollout_val = _rollout_value(env_sim, model, root_player)
        estimate = 0.5 * rollout_val + 0.5 * root_value

        n[action] += 1.0
        q[action] += (estimate - q[action]) / n[action]

    best = valid[np.argmax(n[valid])]
    return int(best)


def agent_action(model, env, mcts_sims: int = 64) -> int:
    """Pick an action using MCTS-guided search backed by policy/value priors."""
    if mcts_sims > 0:
        return mcts_action(env, model, sims=mcts_sims)
    obs = env._get_obs()
    mask = env.action_mask()
    priors, _ = policy_priors_and_value(model, obs, mask)
    action = int(np.argmax(priors))
    if mask[action] <= 0:
        valid = np.where(mask > 0)[0]
        if len(valid) == 0:
            raise RuntimeError("No valid actions available for agent action.")
        return int(valid[0])
    return action


def random_action(env) -> int:
    """Pick a uniformly random valid action."""
    mask = env.action_mask()
    valid = np.where(mask > 0)[0]
    return int(np.random.choice(valid))


def human_action(env, player: int) -> int:
    """Prompt the user to choose a card from their valid options."""
    mask = env.action_mask()
    valid = [c for c in env.hands[player] if mask[c] > 0]

    print(f"\n  Your hand (P{player + 1}):")
    for i, c in enumerate(valid):
        print(f"    [{i + 1}] {card_name(c)}  ({card_short(c)})")

    while True:
        try:
            raw = input(f"  Pick a card [1–{len(valid)}]: ").strip()
            choice = int(raw) - 1
            if 0 <= choice < len(valid):
                return valid[choice]
            print(f"  Enter a number between 1 and {len(valid)}.")
        except ValueError:
            print("  Please enter a number.")


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def choose_mode() -> str:
    """Interactively ask the user to select a game mode."""
    print("\n" + "=" * 50)
    print("  WHIST — Choose a mode:")
    print("  [1] Watch AI play (all 4 seats)")
    print("  [2] Watch AI vs random opponents")
    print("  [3] Play against the AI (you are P1)")
    print("=" * 50)
    mapping = {"1": "watch", "2": "random", "3": "play"}
    while True:
        choice = input("  Enter choice (1 / 2 / 3): ").strip()
        if choice in mapping:
            return mapping[choice]
        print("  Invalid choice – please enter 1, 2, or 3.")


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


def run_game(mode: str, mcts_sims: int) -> None:
    """Run a complete Whist round in the requested mode."""
    model = load_model()
    env = WhistEnv(render_mode=None)  # display is handled here, not in env
    env.reset()

    mode_labels = {
        "watch":  "AI vs AI (all 4 seats)",
        "random": "AI (P1) vs Random opponents (P2/P3/P4)",
        "play":   "You (P1) vs AI (P2/P3/P4)",
    }

    print("\n" + "=" * 50)
    print(f"  WHIST — Trump: {trump_name(env.trump_suit)}")
    print(f"  Team 0: P1 & P3  |  Team 1: P2 & P4")
    print(f"  Mode: {mode_labels[mode]}")
    print("=" * 50)

    trick_num = 0
    trick_display: list = []   # (player, card) pairs for the ongoing trick

    while not env.done:
        player = env.current_player

        # --- Start of a new trick ---
        if len(env.trick_cards) == 0:
            trick_num += 1
            trick_display = []

        # --- Decide action ---
        if mode == "play" and player == 0:
            # Show table state to the human
            if trick_display:
                print("\n  Cards on table so far:")
                for p, c in trick_display:
                    print(f"    P{p + 1}: {card_short(c)}")
            action = human_action(env, player)
            print(f"  → P{player + 1} plays: {card_short(action)}")
        elif mode == "random" and player != 0:
            action = random_action(env)
            print(f"  → P{player + 1} plays: {card_short(action)}")
        else:
            action = agent_action(model, env, mcts_sims=mcts_sims)
            print(f"  → P{player + 1} plays: {card_short(action)}")

        trick_display.append((player, action))

        env.step(action)

        # --- Trick just resolved (4 cards played) ---
        if len(trick_display) == NUM_PLAYERS and len(env.trick_cards) == 0:
            print_trick_summary(trick_num, trick_display)
            trick_winner = env.lead_player  # env sets lead_player = trick winner after resolution
            print(
                f"  ★ P{trick_winner + 1} wins Trick {trick_num}!  "
                f"Score → T0: {env.team_tricks[0]}  T1: {env.team_tricks[1]}"
            )
            if not env.done and mode in ("play", "watch"):
                input("  Press Enter for next trick…")

    # --- Final result ---
    print("\n" + "=" * 50)
    print("  FINAL SCORE")
    print(f"  Team 0 (P1 & P3): {env.team_tricks[0]} tricks")
    print(f"  Team 1 (P2 & P4): {env.team_tricks[1]} tricks")
    if env.team_tricks[0] > env.team_tricks[1]:
        print("  🏆 Team 0 (P1 & P3) wins!")
    elif env.team_tricks[1] > env.team_tricks[0]:
        print("  🏆 Team 1 (P2 & P4) wins!")
    else:
        print("  🤝 It's a tie!")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Play or watch Whist with a trained AI agent."
    )
    parser.add_argument(
        "--mode",
        choices=["watch", "random", "play"],
        help=(
            "watch  – AI controls all 4 seats; "
            "random – AI (P1) vs random opponents; "
            "play   – you play as P1 against the AI."
        ),
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=64,
        help="Number of root MCTS simulations for AI moves (0 disables search).",
    )
    args = parser.parse_args()

    mode = args.mode if args.mode else choose_mode()
    run_game(mode, mcts_sims=max(0, args.mcts_sims))
