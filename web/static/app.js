// Browser client for the Genesis voice agent: chat + hands-free call UI, mic
// capture, VAD/barge-in, and gapless playback of streamed audio.
const $ = (id) => document.getElementById(id);
const player = $("player");

function fmtPrice(p, cur) { return p ? p.toLocaleString("en-US") + " " + (cur || "SAR") : "On request"; }

function makeCard(c, i) {
  const card = document.createElement("div");
  card.className = "card";
  card.style.setProperty("--c", c.accent || "#5a6470");
  card.style.animationDelay = (i * 70) + "ms";
  const miles = c.mileage === 0 ? "New · 0 km" : (c.mileage ? c.mileage.toLocaleString() + " km" : "—");
  const tags = [c.body, c.fuel, c.drivetrain, miles].filter(Boolean).map(t => `<span class="tag">${t}</span>`).join("");
  card.innerHTML = `
    <div class="card-top"><span class="card-badge">${c.year || ""}</span><span class="card-model">${c.model || ""}</span></div>
    <div class="card-body">
      <div class="card-name">${c.fullName || c.name || ""}</div>
      <div class="card-price">${fmtPrice(c.price, c.currency)}</div>
      <div class="card-meta"><span class="tag"><span class="swatch"></span>${c.color || ""}</span>${tags}</div>
    </div>`;
  return card;
}

function renderCars(target, cars) {
  target.innerHTML = "";
  (cars || []).forEach((c, i) => target.appendChild(makeCard(c, i)));
}

function renderFloat(cars) {
  const f = document.getElementById("callFloat"); if (!f) return;
  f.innerHTML = "";
  (cars || []).slice(0, 4).forEach((c, i) => {
    const img = c.image ? c.image + (c.image.includes("?") ? "&" : "?") + "w=380&h=240&fit=crop" : "";
    const tags = [c.fuel, c.drivetrain, ...(c.features || []).slice(0, 2)]
      .filter(Boolean).map(t => `<span class="fc-tag">${t}</span>`).join("");
    const el = document.createElement("div");
    el.className = "float-card fc-" + i;
    el.style.setProperty("--c", c.accent || "#5a6470");
    el.innerHTML = `<div class="fc-anim">
      <div class="fc-img" style="background-image:url('${img}')"><span class="fc-badge">${c.year || ""}</span></div>
      <div class="fc-body">
        <div class="fc-model"><span class="fc-dot"></span>${c.model || ""}</div>
        <div class="fc-price">${fmtPrice(c.price, c.currency)}</div>
        <div class="fc-tags">${tags}</div>
      </div></div>`;
    f.appendChild(el);
  });
}

function downsampleTo16k(samples, inRate) {
  if (inRate === 16000) return floatToI16(samples);
  const ratio = inRate / 16000, outLen = Math.floor(samples.length / ratio), out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio, lo = Math.floor(idx), hi = Math.min(lo + 1, samples.length - 1);
    out[i] = samples[lo] + (samples[hi] - samples[lo]) * (idx - lo);
  }
  return floatToI16(out);
}
function floatToI16(f) {
  const o = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) { const s = Math.max(-1, Math.min(1, f[i])); o[i] = s < 0 ? s * 32768 : s * 32767; }
  return o;
}
function pcmToB64(i16) {
  let s = "", b = new Uint8Array(i16.buffer);
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return btoa(s);
}
function wavUrl(b64) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

// Gapless sequential playback: each chunk waits for the previous to end, and its
// sentence text is surfaced via onText exactly when that chunk starts playing.
const AudioQ = {
  q: [], playing: false, onDrain: null, onText: null,
  push(wav, text) { this.q.push({ wav, text }); if (!this.playing) this.next(); },
  next() {
    if (!this.q.length) { this.playing = false; const cb = this.onDrain; this.onDrain = null; if (cb) cb(); return; }
    this.playing = true;
    const item = this.q.shift();
    if (item.text && this.onText) this.onText(item.text);
    player.src = wavUrl(item.wav);
    player.onended = () => this.next();
    player.play().catch(() => this.next());
  },
  reset() { this.q = []; this.playing = false; this.onDrain = null; try { player.pause(); } catch {} },
};
function whenSpeechDone(cb) { if (AudioQ.playing || AudioQ.q.length) AudioQ.onDrain = cb; else cb(); }

function show(id) {
  document.querySelectorAll(".screen").forEach(s => s.classList.toggle("is-active", s.id === id));
}

(function () {
  const wm = $("wordmark");
  "GENESIS".split("").forEach((ch, i) => {
    const s = document.createElement("span"); s.textContent = ch; s.style.animationDelay = (0.5 + i * 0.06) + "s"; wm.appendChild(s);
  });
})();

let ws, active = null, ready = false;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => route(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1500);
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function route(m) {
  if (m.type === "loading") { $("splashSub").textContent = "Warming up the concierge…"; return; }
  if (m.type === "ready") {
    if (m.build) { console.log("Genesis Concierge " + m.build); let b = document.getElementById("buildTag"); if (!b) { b = document.createElement("div"); b.id = "buildTag"; b.style.cssText = "position:fixed;bottom:6px;right:10px;font:10px monospace;color:#5d5d66;z-index:99;pointer-events:none"; document.body.appendChild(b); } b.textContent = m.build; }
    if (ready) return; ready = true;
    $("splashSub").textContent = "Ready";
    setTimeout(() => show("mode"), 650);
    return;
  }
  if (m.type === "end_call") {                 // end only after the farewell finishes playing
    whenSpeechDone(() => { if (active === CallUI) CallUI.end(); else show("mode"); });
    return;
  }
  if (active && active["on_" + m.type]) active["on_" + m.type](m);
}

document.querySelectorAll("[data-mode]").forEach(card => {
  card.addEventListener("click", () => {
    const mode = card.dataset.mode;
    if (mode === "chat") { active = ChatUI; ChatUI.enter(); }
    else { active = CallUI; CallUI.enter(); }
  });
});

const ChatUI = {
  bubble: null, busy: false,
  enter() { show("chat"); AudioQ.onText = null; send({ type: "reset" }); this.setStatus("Ready", ""); $("textInput").focus(); },
  setStatus(t, cls) { $("statusText").textContent = t; $("statusDot").className = "dot" + (cls ? " " + cls : ""); },
  addMsg(who, text) {
    const w = $("chat").querySelector(".welcome"); if (w) w.style.display = "none";
    const el = document.createElement("div"); el.className = "msg " + who;
    el.innerHTML = `<span class="who">${who === "user" ? "You" : "Aria"}</span><span></span>`;
    el._t = el.lastChild; el._t.textContent = text;
    $("conversation").appendChild(el); this.scroll(); return el;
  },
  scroll() { const c = $("conversation"); c.scrollTop = c.scrollHeight; },
  on_transcript(m) { if (m.text) this.addMsg("user", m.text); this.setStatus("Thinking…", "busy"); },
  on_cars(m) { renderCars($("cars"), m.cars); },
  on_reply_start() { this.bubble = this.addMsg("aria", ""); this.bubble.dataset.full = ""; const c = document.createElement("span"); c.className = "cursor"; this.bubble.appendChild(c); this.bubble._cur = c; this.setStatus("Responding…", "busy"); },
  on_reply_delta(m) { if (!this.bubble) return; this.bubble.dataset.full += m.text; this.bubble._t.textContent = this.bubble.dataset.full; this.scroll(); },
  on_reply_end() { if (this.bubble && this.bubble._cur) this.bubble._cur.remove(); this.bubble = null; },
  on_audio(m) { this.setStatus("Speaking…", "busy"); AudioQ.push(m.wav, m.text); },
  on_speech_end() { whenSpeechDone(() => { this.setStatus("Ready", ""); this.busy = false; }); },
  on_error(m) { this.addMsg("aria", "⚠ " + m.text); this.setStatus("Ready", ""); this.busy = false; },
  sendText(text) {
    text = (text || $("textInput").value).trim();
    if (!text || this.busy) return;
    this.addMsg("user", text); send({ type: "text", text, mode: "chat" }); $("textInput").value = "";
    this.busy = true; this.setStatus("Thinking…", "busy");
  },
};
$("sendBtn").onclick = () => ChatUI.sendText();
$("textInput").addEventListener("keydown", e => { if (e.key === "Enter") ChatUI.sendText(); });
document.querySelectorAll("[data-q]").forEach(ch => ch.addEventListener("click", () => ChatUI.sendText(ch.dataset.q)));
$("chatBack").onclick = () => show("mode");

let chatRec = null;
$("micBtn").addEventListener("click", async () => {
  if (ChatUI.busy && !chatRec) return;
  if (chatRec) { chatRec.stop(); return; }
  chatRec = await SimpleRecorder.start(() => {}, (pcm16) => {
    chatRec = null; $("micBtn").classList.remove("rec");
    if (pcm16.length < 1600) { ChatUI.setStatus("Ready", ""); return; }
    send({ type: "audio", rate: 16000, pcm: pcmToB64(pcm16), mode: "chat" });
    ChatUI.busy = true; ChatUI.setStatus("Transcribing…", "busy");
  });
  if (chatRec) { $("micBtn").classList.add("rec"); ChatUI.setStatus("Listening…", "rec"); }
});

// Push-to-talk recorder shared by chat mic; buffers float frames until stop().
const SimpleRecorder = {
  async start(onLevel, onDone) {
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
    catch { return null; }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(4096, 1, 1);
    const frames = [];
    proc.onaudioprocess = e => { const d = e.inputBuffer.getChannelData(0); frames.push(new Float32Array(d)); };
    src.connect(proc); proc.connect(ctx.destination);
    return {
      stop() {
        proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop());
        const rate = ctx.sampleRate; ctx.close();
        let len = frames.reduce((a, c) => a + c.length, 0), flat = new Float32Array(len), o = 0;
        for (const c of frames) { flat.set(c, o); o += c.length; }
        onDone(downsampleTo16k(flat, rate));
      }
    };
  }
};

// Hands-free call mode: client-side VAD decides when the user stopped talking.
const VAD = { speechThr: 0.015, silenceMs: 1000, maxMs: 13000, bargeThr: 0.05, bargeFrames: 3, bargeGraceMs: 500 };

const CallUI = {
  state: "idle", frames: [], hadSpeech: false, silence: 0, t0: 0, muted: false, caption: "", secs: 0,
  draining: false, bargeCount: 0,
  async enter() {
    show("call");
    this.draining = false; this.bargeCount = 0;
    document.getElementById("callFloat").innerHTML = "";
    // reveal captions in sync with speech: append each sentence as its audio starts
    AudioQ.onText = (txt) => { this.caption = (this.caption + " " + txt).trim(); this.setCaption(this.caption); };
    this.setState("thinking"); this.setStatus("Connecting…"); this.caption = ""; this.setCaption("");
    this.secs = 0; $("callTimer").textContent = "00:00";
    this.timer = setInterval(() => { this.secs++; const m = String(this.secs / 60 | 0).padStart(2, "0"), s = String(this.secs % 60).padStart(2, "0"); $("callTimer").textContent = `${m}:${s}`; }, 1000);
    const ok = await this.openMic();
    if (!ok) { this.setStatus("Microphone blocked"); return; }
    send({ type: "start_call" });
  },
  async openMic() {
    try { this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
    catch { return false; }
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.resume();
    this.src = this.ctx.createMediaStreamSource(this.stream);
    this.proc = this.ctx.createScriptProcessor(2048, 1, 1);
    this.proc.onaudioprocess = e => this.onFrame(e.inputBuffer.getChannelData(0));
    this.src.connect(this.proc); this.proc.connect(this.ctx.destination);
    return true;
  },
  onFrame(buf) {
    let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    const rms = Math.sqrt(s / buf.length);
    // While Aria speaks, listen for barge-in. The grace window keeps her own
    // voice (on open speakers) from instantly tripping the barge detector.
    if (this.state === "speaking") {
      this.setLevel(0.1 + Math.random() * 0.12);
      const pastGrace = performance.now() - this.speakStart > VAD.bargeGraceMs;
      if (!this.muted && pastGrace) {
        if (rms > VAD.bargeThr) { if (++this.bargeCount >= VAD.bargeFrames) this.barge(); }
        else this.bargeCount = 0;
      }
      return;
    }
    this.setLevel(this.state === "listening" ? rms : 0);
    if (this.state !== "listening" || this.muted) return;
    this.frames.push(new Float32Array(buf));
    if (rms > VAD.speechThr) { this.hadSpeech = true; this.silence = 0; }
    else if (this.hadSpeech) this.silence += buf.length / this.ctx.sampleRate * 1000;
    const elapsed = performance.now() - this.t0;
    if ((this.hadSpeech && this.silence >= VAD.silenceMs) || elapsed > VAD.maxMs) this.endUtterance();
  },
  barge() {
    this.bargeCount = 0;
    AudioQ.reset();               // stop Aria immediately
    this.draining = true;         // ignore the rest of the interrupted reply's audio
    this.startListening();
  },
  startListening() {
    this.frames = []; this.hadSpeech = false; this.silence = 0; this.t0 = performance.now(); this.bargeCount = 0;
    this.setState("listening"); this.setStatus("Listening…"); this.setCaption("");
  },
  endUtterance() {
    const had = this.hadSpeech;
    let len = this.frames.reduce((a, c) => a + c.length, 0), flat = new Float32Array(len), o = 0;
    for (const c of this.frames) { flat.set(c, o); o += c.length; }
    this.frames = [];
    if (!had || flat.length < this.ctx.sampleRate * 0.25) { this.startListening(); return; }
    this.setState("thinking"); this.setStatus("Thinking…");
    send({ type: "audio", rate: 16000, pcm: pcmToB64(downsampleTo16k(flat, this.ctx.sampleRate)), mode: "call" });
  },
  on_transcript(m) { this.draining = false; if (m.text) this.setCaption(m.text, true); },
  on_cars(m) { renderFloat(m.cars); },
  on_reply_start() { this.draining = false; this.caption = ""; this.setCaption(""); this.setState("thinking"); this.setStatus("Thinking…"); },
  on_reply_delta() {},   // caption is driven by spoken audio (see AudioQ.onText), not raw text
  on_reply_end() {},
  on_audio(m) {
    if (this.draining) return;   // discard audio from a reply the user interrupted
    if (m.filler) {              // stay in "thinking": mic closed so the filler can't self-trigger barge-in
      this.setStatus("Aria is checking…");
      AudioQ.push(m.wav, m.text);
      return;
    }
    if (this.state !== "speaking") { this.setState("speaking"); this.setStatus("Aria is speaking"); }
    AudioQ.push(m.wav, m.text);
  },
  on_speech_end() {
    if (this.draining) return;   // ignore the interrupted reply's end marker
    whenSpeechDone(() => { this.setLevel(0); if (this.state === "speaking" || this.state === "thinking") this.startListening(); });
  },
  on_error(m) { this.setCaption("⚠ " + m.text); whenSpeechDone(() => this.startListening()); },
  setState(s) {
    this.state = s; const orb = $("orb");
    if (s === "speaking") this.speakStart = performance.now();   // start of barge-in grace
    orb.classList.toggle("listening", s === "listening");
    orb.classList.toggle("speaking", s === "speaking");
    orb.classList.toggle("thinking", s === "thinking");
    // 24s watchdog: if a reply never arrives, recover the stuck call to listening
    clearTimeout(this._wd);
    if (s === "thinking") this._wd = setTimeout(() => {
      if (this.state === "thinking") { AudioQ.reset(); this.startListening(); }
    }, 24000);
  },
  setStatus(t) { $("callStatus").textContent = t; },
  setCaption(t, isUser) {
    const c = $("caption");
    c.textContent = t || ""; c.style.opacity = t ? 1 : 0; c.style.fontStyle = isUser ? "italic" : "normal";
    c.scrollTop = c.scrollHeight;   // keep the newest words in view as she speaks
  },
  setLevel(v) { $("orb").style.setProperty("--level", Math.min(1, v * 12).toFixed(3)); },
  end() {
    if (this.timer) clearInterval(this.timer);
    AudioQ.reset();
    send({ type: "reset" });          // clear the agent's memory when the call ends
    try { this.proc && this.proc.disconnect(); this.src && this.src.disconnect(); this.stream && this.stream.getTracks().forEach(t => t.stop()); this.ctx && this.ctx.close(); } catch {}
    this.state = "idle"; active = null; show("mode");
  },
};
(function () { const b = $("orbBars"); for (let i = 0; i < 5; i++) b.appendChild(document.createElement("i")); })();
$("endBtn").onclick = () => CallUI.end();
$("callBack").onclick = () => CallUI.end();
$("muteBtn").onclick = () => { CallUI.muted = !CallUI.muted; $("muteBtn").classList.toggle("muted", CallUI.muted); };

connect();
