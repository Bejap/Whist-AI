"""Browser demo for playing a human-declarer Esmakker round."""

import json
import os
import threading
import time
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from esmakker_env import BID_ORDER, PASKRIG, EsmakkerGame, NOLO_BIDS
from whist_env import RANKS, SUITS, card_name

HOST = os.getenv("ESMAKKER_WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("ESMAKKER_WEB_PORT", "8001"))
HUMAN_LOG_PATH = os.path.join("graphs", "benchmarks", "esmakker_human_games.jsonl")
HUMAN_LOG_PATH = os.path.join("graphs", "benchmarks", "esmakker_human_games.jsonl")


class EsmakkerWebGame:
    def __init__(self):
        self.lock = threading.Lock()
        self.new_game()

    def new_game(self):
        self.game = EsmakkerGame()
        self.game.dealer = 3
        self.game.current_player = 0
        self.history = []
        self.done = False
        self.message = "Your turn to bid or pass."
        self.display_trick = None
        self.reveal_until = 0.0
        self.game_started_at = time.time()
        self.completed_trick = None
        self.game_started_at = time.time()

    def bid(self, contract):
        if self.game.phase != "bidding" or self.game.current_player != 0:
            raise ValueError("It is not your bidding turn.")
        self.game.bid(0, contract)
        self.history.append({"type": "bid", "player": 0, "bid": contract})
        self._advance_other_bids()
        return self.state()

    def pass_bid(self):
        if self.game.phase != "bidding" or self.game.current_player != 0:
            raise ValueError("It is not your bidding turn.")
        self.game.pass_bid(0)
        self.history.append({"type": "pass", "player": 0})
        self._advance_other_bids()
        return self.state()

    def _advance_other_bids(self):
        # Prototype opponents pass automatically; this keeps the full bidding
        # interaction real while reserving opponent bidding for the AI phase.
        while self.game.phase == "bidding" and self.game.current_player != 0:
            player = self.game.current_player
            self.game.pass_bid(player)
            self.history.append({"type": "pass", "player": player})
        if self.game.phase == "choose_trump":
            self.message = "You won the bidding. Choose trump."
        elif self.game.phase == "play":
            if self.game.current_bid == PASKRIG:
                self.message = "Everyone passed: Paskrig. Take as few tricks as possible."
            else:
                self.message = "Nolo contract. Your turn to play."
        elif self.game.phase == "bidding":
            self.message = "Your turn to bid or pass."

    def choose_trump(self, suit):
        self.game.choose_trump(0, int(suit))
        self.game.phase = "choose_partner"
        self.message = "Choose the partner suit. It cannot be trump."
        return self.state()

    def choose_partner(self, suit):
        self.game.choose_partner_suit(0, int(suit))
        self.message = "Your turn. Play a legal card."
        return self.state()

    def play(self, card):
        if self.game.phase != "play":
            raise ValueError("The game is not in the card-play phase.")
        card = int(card)
        if self.game.current_player != 0:
            raise ValueError("The AI is playing.")
        if card not in self.game.legal_cards(0):
            raise ValueError("That card is not legal.")
        self._play_card(0, card)
        while self.game.phase == "play" and self.game.current_player != 0:
            player = self.game.current_player
            legal = self.game.legal_cards(player)
            action = int(self.game.rng.choice(legal))
            self._play_card(player, action)
        if self.game.phase == "complete":
            self.done = True
            self.message = "Round complete. Start a new deal to play again."
            self._persist_game()
            self._persist_game()
        else:
            self.message = "Your turn."
        return self.state()

    def _play_card(self, player, card):
        trick_before = list(self.game.trick_cards)
        self.game.play_card(player, card)
        self.history.append({
            "type": "card",
            "player": player,
            "card": card,
            "name": card_name(card),
            "trick": list(trick_before) + [(player, card)],
        })
        if len(trick_before) == 3:
            self.display_trick = trick_before + [(player, card)]
            self.reveal_until = time.time() + 1.5

    def _persist_game(self):
        os.makedirs(os.path.dirname(HUMAN_LOG_PATH), exist_ok=True)
        record = {
            "started_at": self.game_started_at,
            "finished_at": time.time(),
            "dealer": self.game.dealer,
            "bid": self.game.current_bid,
            "declarer": self.game.declarer,
            "trump": self.game.trump_suit,
            "partner_revealed": self.game.partner_revealed,
            "tricks_won": self.game.tricks_won,
            "settlement": self.game.round_settlement,
            "history": self.history,
        }
        with open(HUMAN_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=lambda value: value.__dict__) + "\n")
        if len(trick_before) == 3:
            self.completed_trick = {
                "cards": [
                    {"player": p, "card": c, "name": card_name(c)}
                    for p, c in trick_before + [(player, card)]
                ],
                "winner": self.game.lead_player,
            }

    def _persist_game(self):
        os.makedirs(os.path.dirname(HUMAN_LOG_PATH), exist_ok=True)
        record = {
            "started_at": self.game_started_at,
            "finished_at": time.time(),
            "dealer": self.game.dealer,
            "bid": self.game.current_bid,
            "declarer": self.game.declarer,
            "trump": self.game.trump_suit,
            "partner_revealed": self.game.partner_revealed,
            "tricks_won": self.game.tricks_won,
            "settlement": self.game.round_settlement,
            "history": self.history,
        }
        with open(HUMAN_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=lambda value: value.__dict__) + "\n")

    def state(self):
        player_state = self.game.player_state(0)
        visible_trick = self.game.trick_cards
        revealing = self.display_trick is not None and time.time() < self.reveal_until
        if revealing:
            visible_trick = self.display_trick
        elif self.display_trick is not None:
            self.display_trick = None
        return {
            "phase": "reveal" if revealing else self.game.phase,
            "message": self.message,
            "dealer": self.game.dealer,
            "current_player": self.game.current_player,
            "bid": self.game.current_bid,
            "bid_order": list(BID_ORDER),
            "active_bidding": self.game.phase == "bidding" and self.game.current_player == 0,
            "declarer": self.game.declarer,
            "trump": self.game.trump_suit,
            "partner_suit": player_state.get("partner_suit"),
            "partner_revealed": self.game.partner_revealed,
            "hand": [
                {"id": c, "rank": c % 13, "suit": c // 13, "name": card_name(c), "legal": c in player_state["legal_cards"]}
                for c in player_state["hand"]
            ],
            "legal_cards": player_state["legal_cards"],
            "trick": [{"player": p, "card": c, "name": card_name(c)} for p, c in visible_trick],
            "tricks_won": self.game.tricks_won,
            "settlement": self.game.round_settlement,
            "done": self.done,
            "history": self.history[-10:],
            "completed_trick": self.completed_trick,
        }


GAME = EsmakkerWebGame()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data, default=lambda value: value.__dict__).encode()
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_json(INDEX_HTML.encode(), content_type="text/html")
        elif path == "/api/state":
            with GAME.lock:
                self.send_json(GAME.state())
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or "{}")
            with GAME.lock:
                path = urlparse(self.path).path
                if path == "/api/new":
                    GAME.new_game()
                    result = GAME.state()
                elif path == "/api/trump":
                    result = GAME.choose_trump(data["suit"])
                elif path == "/api/partner":
                    result = GAME.choose_partner(data["suit"])
                elif path == "/api/bid":
                    result = GAME.bid(data["bid"])
                elif path == "/api/pass":
                    result = GAME.pass_bid()
                elif path == "/api/play":
                    result = GAME.play(data["card"])
                else:
                    self.send_json({"error": "Not found"}, 404)
                    return
            self.send_json(result)
        except (ValueError, TypeError, KeyError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def log_message(self, format, *args):
        return


INDEX_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Esmakker Whist</title>
<style>
:root{--felt:#153d3b;--felt2:#1c5b53;--ink:#f6f0df;--muted:#b3c9bd;--gold:#ebc26d;--card:#fffaf0;--red:#b23b3b}*{box-sizing:border-box}body{margin:0;background:#081f20;color:var(--ink);font-family:Georgia,serif}.app{max-width:1180px;margin:auto;padding:24px 20px}header{display:flex;justify-content:space-between;align-items:end;margin-bottom:16px}.eyebrow{font:11px Arial;color:var(--gold);letter-spacing:2px;text-transform:uppercase}h1{font-size:clamp(30px,5vw,52px);font-weight:500;margin:6px 0}.status{font:13px Arial;color:var(--muted);text-align:right}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}button,select{border:1px solid #ffffff2b;background:#24564f;color:var(--ink);padding:10px 13px;border-radius:5px;font:600 13px Arial;cursor:pointer}button:hover{background:#317568}button:disabled{opacity:.4;cursor:not-allowed}.table{min-height:480px;border-radius:18px;border:1px solid #31766b;background:radial-gradient(ellipse,var(--felt2),var(--felt) 70%);position:relative;padding:22px;box-shadow:inset 0 0 0 8px #0002}.center{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);min-width:260px;min-height:160px;width:42%;display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;border:1px solid #ffffff1c;border-radius:14px}.trick-card{position:relative}.trick-card small{position:absolute;bottom:-16px;left:50%;transform:translateX(-50%);font:10px Arial;color:var(--muted)}.card{width:62px;height:88px;background:var(--card);color:#1d2c2a;border:1px solid #c9bfae;border-radius:7px;padding:7px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 4px 8px #0004}.card.red{color:var(--red)}.card .suit{font-size:27px;text-align:center}.hand{display:flex;gap:7px;overflow-x:auto;justify-content:center;padding:12px 0}.hand .card{cursor:pointer;flex:0 0 auto}.hand .card:hover{transform:translateY(-8px)}.panel{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.box{background:#102f2d;border:1px solid #ffffff20;border-radius:8px;padding:14px}.box h2{font-size:17px;font-weight:500;margin:0 0 8px}.box p{font:13px Arial;color:var(--muted);line-height:1.45;margin:4px 0}.accent{color:var(--gold)}.history{font:12px Arial;color:var(--muted);max-height:130px;overflow:auto}@media(max-width:700px){header{display:block}.status{text-align:left;margin-top:10px}.table{min-height:430px}.center{width:65%;min-width:0}.panel{grid-template-columns:1fr}.card{width:52px;height:75px}.card .suit{font-size:22px}}
</style></head><body><div class="app"><header><div><div class="eyebrow">Esmakker Whist</div><h1>Play the contract</h1></div><div class="status" id="status">Connecting...</div></header><div class="toolbar"><button onclick="newGame()">New deal</button><span id="phase" class="status"></span></div><section class="table"><div class="center" id="trick"></div></section><div class="hand" id="hand"></div><section class="panel"><div class="box"><h2><span class="accent" id="contract">Contract 9</span></h2><p id="prompt">Choose trump.</p><div id="controls"></div><p>Score: <span id="score">0 - 0 - 0 - 0</span></p></div><div class="box"><h2>Recent play</h2><div class="history" id="history"></div></div></section></div>
<script>const suits=['♣','♦','♥','♠'];const ranks=['2','3','4','5','6','7','8','9','10','J','Q','K','A'];let state;const $=id=>document.getElementById(id);async function api(path,opt={}){let r=await fetch(path,{headers:{'Content-Type':'application/json'},...opt});let d=await r.json();if(!r.ok)throw Error(d.error);return d}function card(c){return `<div class="card ${c.suit==1||c.suit==2?'red':''}"><span>${ranks[c.rank]}</span><span class="suit">${suits[c.suit]}</span><span>${ranks[c.rank]}</span></div>`}function render(s){state=s;$('status').textContent=s.message;$('phase').textContent=`Phase: ${s.phase}`;$('score').textContent=s.tricks_won.join(' - ');$('contract').textContent=`Contract ${s.bid||'No bid'}${s.trump<4?' · Trump '+suits[s.trump]:''}`;$('hand').innerHTML=s.hand.map(c=>`<button class="card ${c.suit==1||c.suit==2?'red':''}" ${!c.legal||s.phase!='play'?'disabled':''} onclick="play(${c.id})"><span>${ranks[c.rank]}</span><span class="suit">${suits[c.suit]}</span><span>${ranks[c.rank]}</span></button>`).join('');$('trick').innerHTML=s.trick.length?s.trick.map(t=>`<div class="trick-card">${card({rank:t.card%13,suit:Math.floor(t.card/13)})}<small>P${t.player+1}</small></div>`).join(''):'<p style="color:var(--muted);font:13px Arial">The trick will appear here.</p>';$('history').innerHTML=s.history.slice().reverse().map(e=>e.type=='bid'?`P${e.player+1} bid <b>${e.bid}</b>`:e.type=='pass'?`P${e.player+1} passed`: `P${e.player+1} played <b>${e.name}</b>`).join('<br>');let controls='';if(s.phase=='bidding'&&s.active_bidding){let current=s.bid?s.bid_order.indexOf(s.bid):-1;controls=s.bid_order.map((bid,i)=>`<button ${i<=current?'disabled':''} onclick="bid('${bid}')">${bid.replace('_',' ')}</button>`).join(' ')+' <button onclick="passBid()">Pass</button>';}if(s.phase=='choose_trump')controls=suits.map((x,i)=>`<button onclick="trump(${i})">${x} ${['Clubs','Diamonds','Hearts','Spades'][i]}</button>`).join(' ');if(s.phase=='choose_partner')controls=suits.map((x,i)=>`<button ${i==s.trump?'disabled':''} onclick="partner(${i})">${x} ${['Clubs','Diamonds','Hearts','Spades'][i]}</button>`).join(' ');$('controls').innerHTML=controls;$('prompt').textContent=s.message;if(s.done&&s.settlement)$('prompt').textContent=`Complete: ${s.settlement.contract_succeeded?'contract made':'contract failed'} · value ${s.settlement.value}`;}async function newGame(){render(await api('/api/new',{method:'POST',body:'{}'}))}async function bid(b){render(await api('/api/bid',{method:'POST',body:JSON.stringify({bid:b})}))}async function passBid(){render(await api('/api/pass',{method:'POST',body:'{}'}))}async function trump(s){render(await api('/api/trump',{method:'POST',body:JSON.stringify({suit:s})}))}async function partner(s){render(await api('/api/partner',{method:'POST',body:JSON.stringify({suit:s})}))}async function play(c){try{render(await api('/api/play',{method:'POST',body:JSON.stringify({card:c})}))}catch(e){alert(e.message)}}api('/api/state').then(render)</script></body></html>'''


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Esmakker web table running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
