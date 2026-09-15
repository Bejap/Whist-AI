import unittest

from whist_ai.engine import GameStateError, WhistGame


class EngineTests(unittest.TestCase):
    def test_seeded_deal_is_reproducible_and_complete(self):
        first = WhistGame(seed=4)
        second = WhistGame(seed=4)
        self.assertEqual(first.hands, second.hands)
        self.assertEqual(sum(len(hand) for hand in first.hands), 52)
        self.assertEqual(len({card for hand in first.hands for card in hand}), 52)

    def test_all_pass_starts_paskrig(self):
        game = WhistGame(seed=1)
        for _ in range(4):
            game.pass_bid(game.current_player)
        self.assertEqual(game.phase, "play")
        self.assertEqual(game.current_bid, "paskrig")
        self.assertIsNone(game.declarer)

    def test_numeric_bid_requires_declaration(self):
        game = WhistGame(seed=2)
        bidder = game.current_player
        game.bid(bidder, 9)
        game.pass_bid(game.current_player)
        game.pass_bid(game.current_player)
        game.pass_bid(game.current_player)
        self.assertEqual(game.phase, "choose_trump")
        self.assertEqual(game.current_player, bidder)
        game.choose_trump(bidder, 0)
        self.assertEqual(game.phase, "choose_partner")
        self.assertNotEqual(game.legal_partner_suits(bidder), (0, 1, 2, 3))

    def test_follow_suit_is_enforced(self):
        game = WhistGame(seed=3)
        bidder = game.current_player
        game.bid(bidder, 7)
        for _ in range(3):
            game.pass_bid(game.current_player)
        game.choose_trump(bidder, 0)
        game.choose_partner_suit(bidder, 1)
        leader = game.current_player
        lead = game.legal_cards(leader)[0]
        game.play_card(leader, lead)
        next_player = game.current_player
        legal = game.legal_cards(next_player)
        suited = [card for card in game.hands[next_player] if card[0] == lead[0]]
        if suited:
            self.assertEqual(set(legal), set(suited))
            with self.assertRaises(GameStateError):
                off_suit = next(card for card in game.hands[next_player] if card[0] != lead[0])
                game.play_card(next_player, off_suit)

    def test_nolo_ends_on_forbidden_trick(self):
        game = WhistGame(seed=5)
        bidder = game.current_player
        game.bid(bidder, "ren sol")
        for _ in range(3):
            game.pass_bid(game.current_player)
        self.assertEqual(game.phase, "play")
        game.tricks_won[bidder] = 1
        other_players = [player for player in range(4) if player != bidder]
        game.trick_cards = [
            (player, game.hands[player][0]) for player in other_players
        ]
        for player, card in game.trick_cards:
            game.hands[player].remove(card)
        game.current_player = bidder
        card = game.legal_cards(bidder)[0]
        game.play_card(bidder, card)
        self.assertEqual(game.phase, "complete")
        self.assertEqual(game.terminal_reward(bidder), -12)


if __name__ == "__main__":
    unittest.main()
