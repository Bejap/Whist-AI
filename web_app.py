"""Local browser UI for playing Whist against the trained agent."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from whist_env import NUM_PLAYERS, TEAMS, WhistEnv, card_name, trump_name

try:
    from play import agent_action, load_model, random_action
except Exception:
    agent_action = load_model = random_action = None

HOST = os.getenv("WHIST_WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("WHIST_WEB_PORT", "8000"))
MCTS_SIMS = max(0, int(os.getenv("WHIST_WEB_MCTS_SIMS", "16")))


class WhistWebGame:
    """Own the game state and advance AI seats after each human move."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_error = None
        self.env = WhistEnv()
        self.new_game()

    def new_game(self):
        self.env.reset()
        self.history = []
        self.message = "Your turn"
        self.done = False
        self.last_completed_trick = None

    def ensure_model(self):
        if self.model is not None or self.model_error is not None:
            return
        try:
            self.model = load_model()
        except Exception as exc:
            self.model_error = str(exc)

    def choose_ai_action(self):
        self.ensure_model()
        if self.model is None:
            valid = np.flatnonzero(self.env.action_mask())
            return int(np.random.choice(valid))
        return int(agent_action(self.model, self.env, mcts_sims=MCTS_SIMS))

    def advance_ai(self):
        while not self.env.done and self.env.current_player != 0:
            player = self.env.current_player
            action = self.choose_ai_action()
            valid_cards = [int(card) for card in np.flatnonzero(self.env.action_mask())]
            trick_before = list(self.env.trick_cards)
            _, reward, terminated, truncated, info = self.env.step(action)
            if not self.env.trick_cards and len(trick_before) == 3:
                self.record_completed_trick(
                    trick_before + [(player, action)], self.env.lead_player
                )
            self.history.append({
                "player": player,
                "action": action,
                "card": card_name(action),
                "valid_cards": valid_cards,
                "reward": float(reward),
                "team_tricks": list(self.env.team_tricks),
            })
            if terminated or truncated:
                self.done = True
                break

    def play(self, action):
        if self.done:
            raise ValueError("This game is over. Start a new game.")
        if self.env.current_player != 0:
            raise ValueError("The AI is still thinking.")
        self.last_completed_trick = None
        action = int(action)
        if action not in self.env.hands[0] or self.env.action_mask()[action] == 0:
            raise ValueError("That card is not legal right now.")
        valid_cards = [int(card) for card in np.flatnonzero(self.env.action_mask())]
        trick_before = list(self.env.trick_cards)
        _, reward, terminated, truncated, info = self.env.step(action)
        if not self.env.trick_cards and len(trick_before) == 3:
            self.record_completed_trick(
                trick_before + [(0, action)], self.env.lead_player
            )
        self.history.append({
            "player": 0,
            "action": action,
            "card": card_name(action),
            "valid_cards": valid_cards,
            "reward": float(reward),
            "team_tricks": list(self.env.team_tricks),
        })
        if terminated or truncated:
            self.done = True
        else:
            self.advance_ai()
        return self.state()

    def record_completed_trick(self, trick_cards, winner):
        self.last_completed_trick = {
            "cards": [
                {"player": int(player), "card": int(card), "name": card_name(card)}
                for player, card in trick_cards
            ],
            "winner": int(winner),
        }

    def state(self):
        legal = self.env.action_mask() if not self.env.done else np.zeros(52)
        return {
            "hand": [
                {
                    "id": int(card),
                    "name": card_name(card),
                    "rank": card % 13,
                    "suit": card // 13,
                    "legal": bool(legal[card] > 0),
                }
                for card in self.env.hands[0]
            ],
            "trick": [
                {"player": int(player), "card": int(card), "name": card_name(card)}
                for player, card in self.env.trick_cards
            ],
            "trump": trump_name(int(self.env.trump_suit)),
            "trump_suit": int(self.env.trump_suit),
            "team_tricks": list(self.env.team_tricks),
            "current_player": int(self.env.current_player),
            "tricks_played": int(self.env.tricks_played),
            "done": bool(self.env.done),
            "winner_team": (
                0 if self.env.team_tricks[0] > self.env.team_tricks[1]
                else 1 if self.env.team_tricks[1] > self.env.team_tricks[0]
                else None
            ) if self.env.done else None,
            "message": self.message,
            "model_loaded": self.model is not None,
            "model_error": self.model_error,
            "mcts_sims": MCTS_SIMS,
            "history": self.history[-8:],
            "completed_trick": self.last_completed_trick,
        }


GAME = WhistWebGame()


class RequestHandler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(INDEX_HTML, content_type="text/html")
        elif path == "/api/state":
            with GAME.lock:
                self._send(json.dumps(GAME.state()))
        else:
            self._send(json.dumps({"error": "Not found"}), 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or "{}")
            with GAME.lock:
                if path == "/api/new":
                    GAME.new_game()
                    result = GAME.state()
                elif path == "/api/play":
                    result = GAME.play(data.get("action"))
                else:
                    self._send(json.dumps({"error": "Not found"}), 404)
                    return
            self._send(json.dumps(result))
        except (ValueError, TypeError, KeyError) as exc:
            self._send(json.dumps({"error": str(exc)}), 400)
        except Exception as exc:
            self._send(json.dumps({"error": str(exc)}), 500)

    def log_message(self, format, *args):
        return


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whist Table</title>
<style>
:root { --felt:#123b38; --felt-2:#18504a; --ink:#f4efe2; --muted:#a9c3b8; --gold:#e8bd68; --red:#b94b4b; --card:#fffaf0; --card-ink:#192c2b; --line:rgba(255,255,255,.14); }
* { box-sizing:border-box; } body { margin:0; color:var(--ink); background:#0b2727; font-family: Georgia, 'Times New Roman', serif; min-height:100vh; }
.app { max-width:1220px; margin:auto; padding:22px clamp(14px,3vw,42px) 30px; }
header { display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:18px; } h1 { margin:0; font-size:clamp(28px,4vw,48px); letter-spacing:0; font-weight:500; } .eyebrow { color:var(--gold); text-transform:uppercase; font:600 11px/1.2 Arial,sans-serif; letter-spacing:2px; margin-bottom:7px; } .status { text-align:right; color:var(--muted); font:13px Arial,sans-serif; }
.toolbar { display:flex; gap:10px; align-items:center; margin-bottom:16px; } button { font:600 13px Arial,sans-serif; border:1px solid var(--line); color:var(--ink); background:#214b47; padding:10px 14px; border-radius:5px; cursor:pointer; } button:hover { background:#2c635c; } button:focus-visible { outline:2px solid var(--gold); outline-offset:3px; }
.table { position:relative; min-height:590px; border:1px solid #286259; border-radius:18px; overflow:hidden; background:radial-gradient(ellipse at center, var(--felt-2), var(--felt) 70%); box-shadow:inset 0 0 0 8px rgba(0,0,0,.08), 0 20px 50px rgba(0,0,0,.25); }
.table:after { content:''; position:absolute; inset:18px; border:1px solid rgba(232,189,104,.28); border-radius:14px; pointer-events:none; }
.seat { position:absolute; display:flex; flex-direction:column; align-items:center; gap:8px; z-index:2; } .seat.top { top:25px; left:50%; transform:translateX(-50%); } .seat.left { left:27px; top:50%; transform:translateY(-50%); } .seat.right { right:27px; top:50%; transform:translateY(-50%); } .seat.bottom { bottom:25px; left:50%; transform:translateX(-50%); }
.seat-label { color:var(--muted); font:600 12px Arial,sans-serif; text-transform:uppercase; letter-spacing:1px; } .seat-score { color:var(--gold); font:14px Arial,sans-serif; }
.pip { display:grid; place-items:center; width:70px; height:54px; border:1px solid rgba(255,255,255,.2); border-radius:9px; background:rgba(255,255,255,.08); color:var(--ink); font:bold 25px Georgia,serif; } .pip.red { color:#e78678; }
.center { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:min(410px,58%); min-height:220px; border:1px solid rgba(255,255,255,.1); border-radius:14px; background:rgba(5,27,27,.18); display:flex; flex-wrap:wrap; justify-content:center; align-content:center; gap:16px; padding:38px; z-index:1; } .trick-card { position:relative; } .trick-card small { position:absolute; left:50%; transform:translateX(-50%); bottom:-17px; color:var(--muted); font:10px Arial,sans-serif; white-space:nowrap; } .trick-card.reveal .card { animation:card-arrive .42s cubic-bezier(.2,.8,.2,1) both; animation-delay:calc(var(--i) * 150ms); } @keyframes card-arrive { from { opacity:0; transform:translateY(-28px) rotate(-5deg) scale(.86); } to { opacity:1; transform:translateY(0) rotate(0) scale(1); } }
.card { width:62px; height:88px; border:1px solid #c9bfae; border-radius:7px; background:var(--card); color:var(--card-ink); box-shadow:0 4px 7px rgba(0,0,0,.2); cursor:pointer; display:flex; flex-direction:column; justify-content:space-between; padding:7px 8px; font:700 17px Georgia,serif; transition:transform .14s, box-shadow .14s, opacity .14s; } .card:hover:not(:disabled) { transform:translateY(-10px); box-shadow:0 12px 15px rgba(0,0,0,.3); } .card:disabled { cursor:not-allowed; opacity:.42; } .card.red { color:#b23434; } .card .corner { font-size:13px; } .card .suit { align-self:center; font-size:27px; } .hand { position:relative; z-index:5; display:flex; justify-content:center; gap:7px; min-height:112px; padding:11px 8px; overflow-x:auto; } .hand .card { flex:0 0 auto; }
.panel { display:grid; grid-template-columns:1.3fr 1fr; gap:14px; margin-top:16px; } .info { border:1px solid var(--line); background:#102f2d; border-radius:8px; padding:15px 17px; } .info h2 { margin:0 0 8px; font-size:16px; font-weight:500; } .info p { margin:4px 0; color:var(--muted); font:13px/1.45 Arial,sans-serif; } .accent { color:var(--gold); } .history { max-height:100px; overflow:auto; color:var(--muted); font:12px/1.5 Arial,sans-serif; }
.overlay { position:absolute; inset:0; display:none; place-items:center; background:rgba(5,22,22,.72); z-index:10; } .overlay.show { display:grid; } .result { text-align:center; } .result h2 { font-size:34px; margin:0 0 8px; } .result p { color:var(--muted); font:14px Arial,sans-serif; }
@media (max-width:700px) { .table { min-height:560px; } .seat.left { left:8px; } .seat.right { right:8px; } .center { width:55%; padding:20px 8px; gap:8px; } .pip { width:48px; height:42px; font-size:19px; } .card { width:51px; height:75px; padding:5px; font-size:14px; } .card .suit { font-size:21px; } .panel { grid-template-columns:1fr; } header { align-items:start; flex-direction:column; } .status { text-align:left; } }
</style>
</head>
<body>
<div class="app">
<header><div><div class="eyebrow">Local Whist Table</div><h1>Play the hand</h1></div><div class="status" id="status">Connecting…</div></header>
<div class="toolbar"><button id="new-game">New deal</button><span class="status" id="engine"></span></div>
<section class="table">
  <div class="seat top"><div class="seat-label">North · AI</div><div class="pip" id="north-pip">—</div><div class="seat-score" id="north-score">0 tricks</div></div>
  <div class="seat left"><div class="seat-label">West · AI</div><div class="pip" id="west-pip">—</div><div class="seat-score" id="west-score">0 tricks</div></div>
  <div class="seat right"><div class="seat-label">East · AI</div><div class="pip" id="east-pip">—</div><div class="seat-score" id="east-score">0 tricks</div></div>
  <div class="center" id="trick"></div>
  <div class="seat bottom"><div class="seat-label">South · You</div><div class="seat-score" id="south-score">0 tricks</div></div>
  <div class="overlay" id="overlay"><div class="result"><h2 id="result-title"></h2><p id="result-text"></p><button id="overlay-new">Deal again</button></div></div>
</section>
<div class="hand" id="hand"></div>
<section class="panel"><div class="info"><h2><span class="accent" id="trump">Trump</span></h2><p id="hint">Choose a legal card from your hand.</p></div><div class="info"><h2>Recent play</h2><div class="history" id="history">No cards played yet.</div></div></section>
</div>
<script>
const suits=['♣','♦','♥','♠']; const ranks=['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
let state=null;
const $=id=>document.getElementById(id);
async function api(path, options={}) { const response=await fetch(path,{headers:{'Content-Type':'application/json'},...options}); const data=await response.json(); if(!response.ok) throw new Error(data.error||'Request failed'); return data; }
function cardLabel(card) { return `${ranks[card.rank]}${suits[card.suit]}`; }
function render(s) {
 state=s; $('status').textContent=s.done ? 'Round complete' : s.current_player===0 ? 'Your turn' : 'AI thinking…'; $('engine').textContent=s.model_loaded ? `MaskablePPO · ${s.mcts_sims} MCTS simulations` : 'Random fallback · no checkpoint loaded'; $('trump').textContent=`Trump: ${s.trump}`;
 $('south-score').textContent=`${s.team_tricks[0]} tricks · Team 0`; $('north-score').textContent=`${s.team_tricks[0]} tricks`; $('west-score').textContent=`${s.team_tricks[1]} tricks`; $('east-score').textContent=`${s.team_tricks[1]} tricks`;
 const showingCompleted = Boolean(s.completed_trick); const visibleTrick = showingCompleted ? s.completed_trick.cards : s.trick;
 $('hand').innerHTML=''; s.hand.forEach(card=>{const b=document.createElement('button'); b.className='card '+([1,2].includes(card.suit)?'red':''); b.disabled=!card.legal||s.done||s.current_player!==0||showingCompleted; b.title=card.name; b.innerHTML=`<span class="corner">${ranks[card.rank]}</span><span class="suit">${suits[card.suit]}</span><span class="corner">${ranks[card.rank]}</span>`; b.onclick=()=>play(card.id); $('hand').appendChild(b);});
 $('trick').innerHTML=visibleTrick.length?visibleTrick.map((t,i)=>`<div class="trick-card ${showingCompleted?'reveal':''}" style="--i:${i}"><div class="card ${[1,2].includes(Math.floor(t.card/13))?'red':''}"><span class="corner">${ranks[t.card%13]}</span><span class="suit">${suits[Math.floor(t.card/13)]}</span><span class="corner">${ranks[t.card%13]}</span></div><small>P${t.player+1}</small></div>`).join(''):'<div style="color:var(--muted);font:13px Arial">Cards played here will appear on the table</div>';
 $('history').innerHTML=s.history.length?s.history.slice().reverse().map(e=>`P${e.player+1} played <strong>${e.card}</strong> · ${e.team_tricks[0]}–${e.team_tricks[1]}`).join('<br>'):'No cards played yet.';
 $('hint').textContent=s.done?'Start a new deal to play again.':showingCompleted?`Trick complete · P${s.completed_trick.winner+1} wins. Next deal in a moment…`:s.current_player===0?'Choose a legal card from your hand.':'The AI is completing the trick.';
 $('overlay').classList.toggle('show',s.done); if(s.done){$('result-title').textContent=s.winner_team===0?'Your team wins':'The AI team wins'; $('result-text').textContent=`Final score: ${s.team_tricks[0]}–${s.team_tricks[1]}`;}
 if(showingCompleted){ setTimeout(()=>{ if(state===s){ render({...s,completed_trick:null}); } }, 1500); }
}
async function play(action){ try { render({...state,current_player:1}); render(await api('/api/play',{method:'POST',body:JSON.stringify({action})})); } catch(e){ $('hint').textContent=e.message; render(await api('/api/state')); } }
async function newGame(){ render(await api('/api/new',{method:'POST',body:'{}'})); }
$('new-game').onclick=newGame; $('overlay-new').onclick=newGame; api('/api/state').then(render).catch(e=>$('status').textContent=e.message);
</script>
</body>
</html>'''


def main():
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"Whist web table running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
