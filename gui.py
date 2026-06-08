"""Pygame graphical frontend for Whist AI.

Human plays as P1 (seat 0, bottom) against three PPO-trained AI opponents.
If no checkpoint exists the AI plays randomly.

Run with:  python gui.py
"""

import os
import sys
import time
from typing import Optional

import numpy as np
import pygame

from whist_env import WhistEnv, SUITS, TEAMS, NUM_CARDS, NUM_PLAYERS, NO_TRUMP

# ---------------------------------------------------------------------------
# Layout & colour constants
# ---------------------------------------------------------------------------

WIN_W, WIN_H = 1024, 768
FPS          = 60

CARD_W, CARD_H = 72, 108
CARD_RADIUS    = 8
HAND_OVERLAP   = 30   # pixels between adjacent face-up cards (bottom hand)
SIDE_OVERLAP   = 22   # pixels between adjacent rotated cards (side hands)

FELT_GREEN  = ( 34,  85,  34)
TABLE_DARK  = ( 22,  58,  22)
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
RED_CARD    = (196,  30,  30)
BLUE_BACK   = ( 30,  50, 180)
BLUE_BACK2  = ( 50,  80, 220)
YELLOW      = (255, 220,   0)
GREY        = (155, 155, 155)
DARK_GREY   = ( 80,  80,  80)
GOLD        = (218, 165,  32)
INFO_TEXT   = (220, 220, 200)
HINT_TEXT   = (255, 230, 100)

# Card suit glyphs & colours (index = suit id: 0=Clubs 1=Diamonds 2=Hearts 3=Spades)
RANK_SHORT  = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUIT_SYMBOL = ["\u2663", "\u2666", "\u2665", "\u2660"]  # ♣ ♦ ♥ ♠
SUIT_COLOR  = [BLACK, RED_CARD, RED_CARD, BLACK]

# Delays
AI_MOVE_DELAY    = 0.5   # seconds between AI moves
TRICK_END_PAUSE  = 1.0   # seconds to show a completed trick

# Offsets from window centre for each player's card in the trick area
#   seat 0 = bottom (human), 1 = left, 2 = top, 3 = right
TRICK_OFFSETS = [
    (  0,  95),   # P1 – bottom
    (-100,  0),   # P2 – left
    (  0, -95),   # P3 – top
    ( 100,  0),   # P4 – right
]

# Seating labels shown near each hand (1-indexed for the user)
SEAT_LABEL = {
    0: "You (P1)",
    1: "P2",
    2: "P3  (partner)",
    3: "P4",
}

# ---------------------------------------------------------------------------
# State tokens
# ---------------------------------------------------------------------------
HUMAN_TURN  = "human"
AI_TURN     = "ai"
TRICK_END   = "trick_end"
ROUND_END   = "round_end"

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

_fonts: dict = {}


def get_font(size: int) -> pygame.font.Font:
    """Return a cached font that supports Unicode suit symbols."""
    if size not in _fonts:
        for name in ("segoeuisymbol", "dejavusans", "freesans", "noto", None):
            try:
                _fonts[size] = (
                    pygame.font.SysFont(name, size)
                    if name else pygame.font.Font(None, size)
                )
                break
            except Exception:
                continue
    return _fonts[size]


def draw_text(surf: pygame.Surface, text: str, x: int, y: int,
              size: int = 18, color=WHITE, center: bool = False) -> None:
    """Blit rendered text onto *surf*; if *center* is True treat (x, y) as centre."""
    s = get_font(size).render(str(text), True, color)
    if center:
        x -= s.get_width()  // 2
        y -= s.get_height() // 2
    surf.blit(s, (x, y))

# ---------------------------------------------------------------------------
# Card drawing helpers
# ---------------------------------------------------------------------------


def draw_card_face(surf: pygame.Surface, card_id: int, x: int, y: int,
                   w: int = CARD_W, h: int = CARD_H,
                   highlighted: bool = False, greyed: bool = False) -> None:
    """Render a face-up card at top-left (x, y)."""
    suit     = card_id // 13
    rank     = card_id % 13
    symbol   = SUIT_SYMBOL[suit]
    c_col    = SUIT_COLOR[suit]
    rank_str = RANK_SHORT[rank]

    if greyed:
        bg, c_col      = (175, 175, 175), (110, 110, 110)
        border_col, bw = DARK_GREY, 1
    elif highlighted:
        bg, border_col, bw = WHITE, YELLOW, 3
    else:
        bg, border_col, bw = WHITE, (150, 150, 150), 1

    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surf, bg,         rect, border_radius=CARD_RADIUS)
    pygame.draw.rect(surf, border_col, rect, bw, border_radius=CARD_RADIUS)

    f_sm = get_font(16)
    f_lg = get_font(28)

    # Top-left: rank above symbol
    r_s  = f_sm.render(rank_str, True, c_col)
    sy_s = f_sm.render(symbol,   True, c_col)
    surf.blit(r_s,  (x + 4, y + 2))
    surf.blit(sy_s, (x + 4, y + 2 + r_s.get_height()))

    # Centre: large suit symbol
    big  = f_lg.render(symbol, True, c_col)
    surf.blit(big, (x + w // 2 - big.get_width()  // 2,
                    y + h // 2 - big.get_height() // 2))

    # Bottom-right: rank + symbol rotated 180°
    r_s2  = pygame.transform.rotate(f_sm.render(rank_str, True, c_col), 180)
    sy_s2 = pygame.transform.rotate(f_sm.render(symbol,   True, c_col), 180)
    surf.blit(sy_s2, (x + w - sy_s2.get_width() - 4,
                      y + h - sy_s2.get_height() - 2))
    surf.blit(r_s2,  (x + w - r_s2.get_width() - 4,
                      y + h - sy_s2.get_height() - r_s2.get_height() - 2))


def _make_card_back_surface(w: int = CARD_W, h: int = CARD_H) -> pygame.Surface:
    """Build the upright face-down card surface (blue diamond pattern)."""
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, BLUE_BACK, (0, 0, w, h), border_radius=CARD_RADIUS)
    pygame.draw.rect(s, (10, 20, 100), (0, 0, w, h), 2, border_radius=CARD_RADIUS)
    for row in range(4):
        for col in range(3):
            dx = 14 + col * (w - 28) // 2
            dy = 14 + row * (h - 28) // 3
            pts = [(dx, dy - 7), (dx + 7, dy), (dx, dy + 7), (dx - 7, dy)]
            pygame.draw.polygon(s, BLUE_BACK2, pts)
    return s


_back_cache: dict = {}


def get_card_back(angle: int = 0) -> pygame.Surface:
    """Return a cached face-down card surface, optionally rotated."""
    if angle not in _back_cache:
        base = _make_card_back_surface()
        _back_cache[angle] = (
            pygame.transform.rotate(base, angle) if angle else base
        )
    return _back_cache[angle]

# ---------------------------------------------------------------------------
# Hand layout helpers
# ---------------------------------------------------------------------------


def bottom_hand_positions(n: int, y0: int) -> list:
    """(x, y) top-left for each of *n* face-up cards at the bottom of the window."""
    total_w = CARD_W + max(0, n - 1) * HAND_OVERLAP
    start_x = WIN_W // 2 - total_w // 2
    return [(start_x + i * HAND_OVERLAP, y0) for i in range(n)]


def top_hand_positions(n: int, y0: int) -> list:
    """(x, y) top-left for each of *n* face-down upright cards at the top."""
    total_w = CARD_W + max(0, n - 1) * HAND_OVERLAP
    start_x = WIN_W // 2 - total_w // 2
    return [(start_x + i * HAND_OVERLAP, y0) for i in range(n)]


def side_hand_positions(n: int, cx: int, cy: int) -> list:
    """(blit_x, blit_y) for *n* rotated face-down cards centred at (cx, cy).

    After a 90° rotation a card surface is CARD_H wide × CARD_W tall.
    Cards are stacked top-to-bottom with SIDE_OVERLAP spacing.
    """
    rot_w = CARD_H  # rotated card width  (108)
    rot_h = CARD_W  # rotated card height ( 72)
    total_h = rot_h + max(0, n - 1) * SIDE_OVERLAP
    start_y = cy - total_h // 2
    blit_x  = cx - rot_w // 2
    return [(blit_x, start_y + i * SIDE_OVERLAP) for i in range(n)]

# ---------------------------------------------------------------------------
# Model / action helpers
# ---------------------------------------------------------------------------


def load_model():
    """Load the latest PPO checkpoint.  Returns (model, episode) or (None, 0)."""
    try:
        from train import latest_checkpoint
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"Could not import training helpers: {e}")
        return None, 0

    ckpt, ep = latest_checkpoint()
    if ckpt is None:
        print("No checkpoint found — AI will play randomly.")
        return None, 0
    try:
        model = PPO.load(ckpt, device="cpu")
        print(f"Loaded checkpoint (episode {ep}): {ckpt}")
        return model, ep
    except Exception as exc:
        print(f"Could not load checkpoint: {exc} — AI will play randomly.")
        return None, 0


def ai_pick(model, obs: np.ndarray, mask: np.ndarray) -> int:
    """Choose an AI action using the PPO model (same logic as train.py sample_action)."""
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        raise RuntimeError("No valid actions available.")
    if model is None:
        return int(np.random.choice(valid))
    action, _ = model.predict(obs, deterministic=False)
    action = int(action)
    if mask[action] > 0:
        return action
    return int(np.random.choice(valid))

# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------


class WhistGUI:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.model, self.ckpt_ep = load_model()
        self._new_round()

    # ── Round lifecycle ───────────────────────────────────────────────────

    def _new_round(self) -> None:
        self.env = WhistEnv()
        self.env.reset()
        self.trick_display: list = []   # (player_idx, card_id) accumulates each trick
        self.ai_fire_at:    float = 0.0
        self.trick_end_at:  float = 0.0
        self._update_state()

    def _update_state(self) -> None:
        """Derive the next game state from env.current_player / env.done."""
        if self.env.done:
            self.state = ROUND_END
        elif self.env.current_player == 0:
            self.state = HUMAN_TURN
        else:
            self.state = AI_TURN
            self.ai_fire_at = time.time() + AI_MOVE_DELAY

    # ── Card play ─────────────────────────────────────────────────────────

    def _play(self, card: int) -> None:
        """Apply *card* for the current player and update display state."""
        player      = self.env.current_player
        prev_tricks = self.env.tricks_played

        self.env.step(card)
        self.trick_display.append((player, card))

        trick_resolved = self.env.tricks_played > prev_tricks

        if self.env.done:
            # Show final trick briefly before the overlay
            self.state       = TRICK_END
            self.trick_end_at = time.time() + TRICK_END_PAUSE
        elif trick_resolved:
            self.state       = TRICK_END
            self.trick_end_at = time.time() + TRICK_END_PAUSE
        else:
            self._update_state()

    # ── Update (called every frame) ───────────────────────────────────────

    def update(self) -> None:
        now = time.time()

        if self.state == AI_TURN and now >= self.ai_fire_at:
            obs  = self.env._get_obs()
            mask = self.env.action_mask()
            self._play(ai_pick(self.model, obs, mask))

        elif self.state == TRICK_END and now >= self.trick_end_at:
            self.trick_display = []
            if self.env.done:
                self.state = ROUND_END
            else:
                self._update_state()

    # ── Input handling ────────────────────────────────────────────────────

    def handle_event(self, ev: pygame.event.Event) -> None:
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return
        pos = ev.pos

        if self.state == HUMAN_TURN:
            card = self._human_card_at(pos)
            if card is not None and self.env.action_mask()[card] > 0:
                self._play(card)

        elif self.state == ROUND_END:
            if self._play_again_rect().collidepoint(pos):
                self._new_round()

    def _human_card_at(self, pos) -> Optional[int]:
        """Return the card index under *pos* in the human's hand, or None."""
        hand = self.env.hands[0]
        if not hand:
            return None
        y0   = WIN_H - CARD_H - 14
        poss = bottom_hand_positions(len(hand), y0)
        # Iterate reversed so the visually top card gets priority
        for i in reversed(range(len(hand))):
            if pygame.Rect(poss[i][0], poss[i][1], CARD_W, CARD_H).collidepoint(pos):
                return hand[i]
        return None

    # ── Master draw ───────────────────────────────────────────────────────

    def draw(self) -> None:
        s = self.screen
        s.fill(FELT_GREEN)
        self._draw_table_oval()
        self._draw_hud()
        self._draw_trick_area()
        self._draw_trump_indicator()
        self._draw_all_hands()
        if self.state == ROUND_END:
            self._draw_round_overlay()
        pygame.display.flip()

    # ── Background ────────────────────────────────────────────────────────

    def _draw_table_oval(self) -> None:
        cx, cy = WIN_W // 2, WIN_H // 2
        r = pygame.Rect(cx - 310, cy - 230, 620, 460)
        pygame.draw.ellipse(self.screen, TABLE_DARK, r)
        pygame.draw.ellipse(self.screen, (18, 48, 18), r, 3)

    # ── HUD (scores, trick counter, turn prompt) ──────────────────────────

    def _draw_hud(self) -> None:
        cx, cy = WIN_W // 2, WIN_H // 2
        t0, t1 = self.env.team_tricks
        tp     = self.env.tricks_played

        draw_text(self.screen,
                  f"Team 0 (You & P3): {t0}   |   Team 1 (P2 & P4): {t1}",
                  cx, cy - 190, 19, INFO_TEXT, center=True)
        draw_text(self.screen,
                  f"Trick {min(tp + 1, 13)} of 13",
                  cx, cy - 164, 17, GREY, center=True)

        if self.state not in (ROUND_END,):
            cp  = self.env.current_player
            lbl = "Your turn — click a card" if cp == 0 else f"P{cp + 1} thinking…"
            draw_text(self.screen, lbl, cx, cy + 192, 17, HINT_TEXT, center=True)

    # ── Trump indicator (top-right corner) ───────────────────────────────

    def _draw_trump_indicator(self) -> None:
        ts = self.env.trump_suit
        if ts == NO_TRUMP:
            lbl, col = "NT", WHITE
        else:
            lbl, col = SUIT_SYMBOL[ts], SUIT_COLOR[ts]
        draw_text(self.screen, "Trump:", WIN_W - 116, 10, 17, GREY)
        draw_text(self.screen, lbl,     WIN_W -  54,  8, 30, col)

    # ── Trick area (4 played cards around centre) ─────────────────────────

    def _draw_trick_area(self) -> None:
        cx, cy = WIN_W // 2, WIN_H // 2
        for player, card in self.trick_display:
            dx, dy = TRICK_OFFSETS[player]
            tx = cx + dx - CARD_W // 2
            ty = cy + dy - CARD_H // 2
            draw_card_face(self.screen, card, tx, ty)
            lbl = "You" if player == 0 else f"P{player + 1}"
            draw_text(self.screen, lbl,
                      cx + dx, cy + dy + CARD_H // 2 + 5,
                      15, GREY, center=True)

    # ── All four hands ────────────────────────────────────────────────────

    def _draw_all_hands(self) -> None:
        mouse = pygame.mouse.get_pos()
        mask  = (self.env.action_mask()
                 if self.state == HUMAN_TURN
                 else np.zeros(NUM_CARDS, dtype=np.float32))

        self._draw_bottom_hand(mouse, mask)
        self._draw_top_hand()
        self._draw_left_hand()
        self._draw_right_hand()

    def _draw_bottom_hand(self, mouse_pos, mask: np.ndarray) -> None:
        """P1 (human) — face-up, clickable."""
        hand = self.env.hands[0]
        y0   = WIN_H - CARD_H - 14
        poss = bottom_hand_positions(len(hand), y0)

        for i, card in enumerate(hand):
            x, y  = poss[i]
            valid = mask[card] > 0
            hov   = pygame.Rect(x, y, CARD_W, CARD_H).collidepoint(mouse_pos)
            lit   = hov and valid and self.state == HUMAN_TURN
            draw_card_face(self.screen, card, x, y - (14 if lit else 0),
                           highlighted=lit, greyed=not valid)

        # Seat label
        draw_text(self.screen, SEAT_LABEL[0],
                  WIN_W // 2, WIN_H - CARD_H - 32, 15, GREY, center=True)

    def _draw_top_hand(self) -> None:
        """P3 (partner) — face-down upright cards."""
        hand  = self.env.hands[2]
        y0    = 14
        poss  = top_hand_positions(len(hand), y0)
        back  = get_card_back(0)
        for x, y in poss:
            self.screen.blit(back, (x, y))
        draw_text(self.screen, SEAT_LABEL[2],
                  WIN_W // 2, y0 + CARD_H + 6, 15, GREY, center=True)

    def _draw_left_hand(self) -> None:
        """P2 — face-down, cards rotated 90° counter-clockwise."""
        hand    = self.env.hands[1]
        cx      = 14 + CARD_H // 2   # horizontal centre of the left strip
        cy      = WIN_H // 2
        poss    = side_hand_positions(len(hand), cx, cy)
        back90  = get_card_back(90)
        for bx, by in poss:
            self.screen.blit(back90, (bx, by))
        # Label to the right of the left-hand column
        draw_text(self.screen, SEAT_LABEL[1],
                  14 + CARD_H + 6, WIN_H // 2, 15, GREY, center=False)

    def _draw_right_hand(self) -> None:
        """P4 — face-down, cards rotated 90° clockwise."""
        hand     = self.env.hands[3]
        cx       = WIN_W - 14 - CARD_H // 2
        cy       = WIN_H // 2
        poss     = side_hand_positions(len(hand), cx, cy)
        back_n90 = get_card_back(-90)
        for bx, by in poss:
            self.screen.blit(back_n90, (bx, by))
        # Label to the left of the right-hand column
        lbl_w = get_font(15).size(SEAT_LABEL[3])[0]
        draw_text(self.screen, SEAT_LABEL[3],
                  WIN_W - 14 - CARD_H - 6 - lbl_w, WIN_H // 2, 15, GREY)

    # ── Round-end overlay ─────────────────────────────────────────────────

    def _play_again_rect(self) -> pygame.Rect:
        return pygame.Rect(WIN_W // 2 - 100, WIN_H // 2 + 70, 200, 48)

    def _draw_round_overlay(self) -> None:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((10, 25, 10, 215))
        self.screen.blit(overlay, (0, 0))

        cx, cy = WIN_W // 2, WIN_H // 2
        t0, t1 = self.env.team_tricks

        if t0 > t1:
            headline, hcol = "Team 0 Wins!  (You & P3)", GOLD
        elif t1 > t0:
            headline, hcol = "Team 1 Wins  (P2 & P4)", (230, 80, 80)
        else:
            headline, hcol = "It's a Tie!", WHITE

        draw_text(self.screen, headline, cx, cy - 60, 36, hcol, center=True)
        draw_text(self.screen,
                  f"Team 0: {t0} tricks   |   Team 1: {t1} tricks",
                  cx, cy, 24, WHITE, center=True)

        btn   = self._play_again_rect()
        mouse = pygame.mouse.get_pos()
        bcol  = (55, 130, 55) if btn.collidepoint(mouse) else (35, 90, 35)
        pygame.draw.rect(self.screen, bcol,  btn, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn, 2, border_radius=10)
        draw_text(self.screen, "Play Again",
                  btn.centerx, btn.centery, 22, WHITE, center=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Whist AI")
    clock = pygame.time.Clock()

    game = WhistGUI(screen)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(ev)

        game.update()
        game.draw()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
