/* FraudSentinel dashboard - connects to /ws for live data and /api/* for actions */
(() => {
  'use strict';

  /* ---------------------------------------------------------------- helpers */
  const FILE = location.protocol === 'file:';
  const API = FILE ? 'http://localhost:8000' : '';
  const WS_URL = FILE ? 'ws://localhost:8000/ws'
    : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const nf = new Intl.NumberFormat('en-IN');
  const inrFull = v => '₹' + nf.format(Math.round(v));
  const inr = v => {
    v = Math.round(v);
    if (v >= 1e7) return '₹' + (v / 1e7).toFixed(2) + ' Cr';
    if (v >= 1e5) return '₹' + (v / 1e5).toFixed(2) + ' L';
    return '₹' + nf.format(v);
  };
  const clock = ms => new Date(ms).toLocaleTimeString('en-GB', { hour12: false });
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  };
  const REASON_NAMES = {
    VELOCITY: 'Rapid-fire payments', CARD_TESTING: 'Card testing', IMPOSSIBLE_TRAVEL: 'Impossible travel',
    GEO_JUMP: 'Location jump', AMOUNT_SPIKE: 'Unusual amount', RISKY_MERCHANT: 'Risky merchant',
    NEW_DEVICE: 'New device', FOREIGN: 'Foreign payment', ODD_HOURS: 'Odd hours', ML_ANOMALY: 'Model outlier',
  };
  const PALETTE = ['#e0361b', '#e39a0b', '#0d9aa6', '#5b6cf0', '#2fa36b', '#8a6fd6', '#d6568f', '#7a8b99', '#b5651d', '#3e8ede'];
  const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const hexA = (hex, a) => {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
    return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
  };

  // When the server runs with FRAUD_API_KEY, state-changing calls return 401; ask once and remember the key.
  let apiKey = '';
  try { apiKey = localStorage.getItem('fs-api-key') || ''; } catch { /* storage unavailable */ }

  async function api(path, method = 'GET', body, retried = false) {
    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers['X-API-Key'] = apiKey;
    const r = await fetch(API + path, { method, headers, body: body ? JSON.stringify(body) : undefined });
    if (r.status === 401 && !retried) {
      const k = window.prompt('This server requires an API key (FRAUD_API_KEY):');
      if (k) {
        apiKey = k.trim();
        try { localStorage.setItem('fs-api-key', apiKey); } catch { /* ignore */ }
        return api(path, method, body, true);
      }
    }
    if (!r.ok) {
      let d = r.statusText;
      try { const j = await r.json(); d = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* ignore */ }
      throw new Error(d);
    }
    return r.json();
  }

  /* ------------------------------------------------------------------ state */
  let txns = [];                 // newest first
  const alerts = new Map();      // id -> alert
  let alertOrder = [];           // newest first
  let stats = null;
  let config = { rate: 6, paused: false, threshold: 0.65, fraud_rate: 0.04 };
  let feedMode = 'all', alertMode = 'open', query = '';
  let soundOn = false, audioCtx = null, lastToast = 0, lastRenderedTop = null;
  const spark = { total: [], flagged: [], rate: [], money: [], det: [], open: [] };
  let lastSpark = 0;

  /* ------------------------------------------------------------- theme/sound */
  function setTheme(t) {
  document.documentElement.dataset.theme = t;
  if ($('#btnTheme')) $('#btnTheme').textContent = t === 'dark' ? 'Light mode' : 'Dark mode';
  store.set('fs-theme', t);
  restyleCharts();
  setTiles();
}
if ($('#btnTheme')) {
  $('#btnTheme').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
}
  $('#btnSound').onclick = e => {
    soundOn = !soundOn;
    e.currentTarget.setAttribute('aria-pressed', soundOn);
    e.currentTarget.textContent = soundOn ? 'Alert sound on' : 'Alert sound off';
    if (soundOn) beep(660, 0.08);
  };
  function beep(freq = 880, dur = 0.15) {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.frequency.value = freq; g.gain.value = 0.06;
      o.connect(g); g.connect(audioCtx.destination);
      o.start(); o.stop(audioCtx.currentTime + dur);
    } catch { /* ignore */ }
  }

  function toast(title, text, kind = '', onClick) {
    const el = document.createElement('div');
    el.className = 'toast ' + kind;
    el.innerHTML = `<b>${esc(title)}</b>${esc(text)}`;
    if (onClick) el.onclick = onClick;
    $('#toasts').appendChild(el);
    setTimeout(() => el.remove(), 5000);
    while ($('#toasts').children.length > 3) $('#toasts').firstChild.remove();
  }

  /* ------------------------------------------------------------ KPI numbers */
  function setNum(el, val, fmt) {
    const from = el._v ?? val;
    el._v = val;
    cancelAnimationFrame(el._raf);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || from === val) { el.textContent = fmt(val); return; }
    const t0 = performance.now();
    const step = now => {
      const k = Math.min(1, (now - t0) / 350), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (val - from) * e);
      if (k < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }
  function drawSpark(svg, arr) {
    if (!svg) return;
    if (arr.length < 2) { svg.innerHTML = ''; return; }
    const max = Math.max(...arr), min = Math.min(...arr), span = max - min || 1;
    const pts = arr.map((v, i) => [i / (arr.length - 1) * 100, 26 - (v - min) / span * 24]);
    const line = pts.map(p => p.join(',')).join(' ');
    svg.innerHTML = `<polygon points="0,28 ${line} 100,28" fill="currentColor" opacity="0.1"/>` +
      `<polyline points="${line}" fill="none" stroke="currentColor" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;
  }
  function kpi(id, val, fmt, sub) {
    const root = $('#' + id);
    setNum($('.k-val', root), val, fmt);
    $('.k-sub', root).textContent = sub;
  }

  function applyStats(s) {
    stats = s;
    const sc = s.status_counts || {}, sev = s.severity || {};
    kpi('k-total', s.total, v => nf.format(Math.round(v)), `${s.tps.toFixed(1)} per second right now`);
    kpi('k-flagged', s.flagged, v => nf.format(Math.round(v)), `${sev.critical || 0} critical, ${sev.high || 0} high`);
    kpi('k-rate', s.fraud_rate, v => v.toFixed(2) + '%', `${nf.format(s.flagged)} of ${nf.format(s.total)} flagged`);
    kpi('k-money', s.money_at_risk, inr, `of ${inr(s.money_total)} processed`);
    kpi('k-det', s.avg_detection_ms, v => v.toFixed(1) + ' ms', `95th percentile ${s.p95_detection_ms.toFixed(1)} ms`);
    kpi('k-open', sc.open || 0, v => nf.format(Math.round(v)), `${sc.confirmed || 0} confirmed, ${sc.dismissed || 0} marked safe`);

    const now = Date.now();
    if (now - lastSpark > 1000) {
      lastSpark = now;
      const push = (k, v) => { spark[k].push(v); if (spark[k].length > 40) spark[k].shift(); };
      push('total', s.total); push('flagged', s.flagged); push('rate', s.fraud_rate);
      push('money', s.money_at_risk); push('det', s.avg_detection_ms); push('open', sc.open || 0);
      drawSpark($('#k-total .spark'), spark.total); drawSpark($('#k-flagged .spark'), spark.flagged);
      drawSpark($('#k-rate .spark'), spark.rate); drawSpark($('#k-money .spark'), spark.money);
      drawSpark($('#k-det .spark'), spark.det); drawSpark($('#k-open .spark'), spark.open);
    }

    // detection quality panel
    const p = s.performance, pct = v => v == null ? '-' : (v * 100).toFixed(0) + '%';
    $('#pf-p').textContent = pct(p.precision); $('#pf-r').textContent = pct(p.recall); $('#pf-f').textContent = pct(p.f1);
    $('#pf-note').textContent = `Caught ${p.tp}, missed ${p.fn}, false alarms ${p.fp}. Scored against the simulator's hidden fraud labels.`;
    $('#lt-avg').textContent = s.avg_detection_ms.toFixed(1) + ' ms';
    $('#lt-p95').textContent = s.p95_detection_ms.toFixed(1) + ' ms';
    $('#lt-tps').textContent = s.tps.toFixed(1) + ' payments per second';
    const up = s.uptime_s; $('#lt-up').textContent = up >= 3600 ? `${Math.floor(up / 3600)} h ${Math.floor(up % 3600 / 60)} min` : `${Math.floor(up / 60)} min ${up % 60} s`;
    const li = (a, b) => `<li><span>${a}</span><span>${b}</span></li>`;
    $('#topMerch').innerHTML = s.top_merchants.length
      ? s.top_merchants.map(m => li(esc(m.merchant), `${m.count} flagged, ${inr(m.amount)}`)).join('')
      : '<li class="muted">No flagged merchants yet</li>';
    $('#topUsers').innerHTML = s.top_users.length
      ? s.top_users.map(u => li(`${esc(u.name)} <small class="muted">${esc(u.user_id)}</small>`, `${u.count} flagged, ${inr(u.amount)}`)).join('')
      : '<li class="muted">No flagged customers yet</li>';
    updateCharts(s);
  }

  function applyConfig(c) {
    if (!c) return;
    config = c;
    const active = document.activeElement && document.activeElement.type === 'range' ? document.activeElement.id : null;
    if (active !== 'rate') $('#rate').value = c.rate;
    if (active !== 'thr') $('#thr').value = c.threshold;
    if (active !== 'fr') $('#fr').value = Math.round(c.fraud_rate * 100);
    $('#v-rate').textContent = Math.round(c.rate);
    $('#v-thr').textContent = (+c.threshold).toFixed(2);
    $('#v-fr').textContent = Math.round(c.fraud_rate * 100) + '%';
    $('#btnPause').textContent = c.paused ? 'Resume stream' : 'Pause stream';
    setConn(c.paused ? 'paused' : 'live');
  }

  /* ------------------------------------------------------------- connection */
  function setConn(state) {
    const el = $('#conn');
    el.dataset.state = state;
    $('b', el).textContent = { live: 'Live', paused: 'Paused', down: 'Disconnected, retrying', connecting: 'Connecting' }[state];
  }

  /* -------------------------------------------------------------- feed view */
  function noteFor(t) {
    return t.reasons.filter(r => r.code !== 'ML_ANOMALY').slice(0, 2).map(r => `${r.title}: ${r.detail}`).join('; ')
      || (t.reasons[0] ? `${t.reasons[0].title}: ${t.reasons[0].detail}` : 'Flagged by the model');
  }
  function matches(t) {
    if (feedMode === 'flagged' && !t.flagged) return false;
    if (!query) return true;
    return `${t.user_name} ${t.user_id} ${t.merchant} ${t.city} ${t.category} ${t.id}`.toLowerCase().includes(query);
  }
  function renderFeed() {
    const list = txns.filter(matches).slice(0, 70);
    if (!list.length) { $('#feed').innerHTML = '<p class="empty">Nothing matches yet.</p>'; return; }
    let reachedOld = false;
    const html = list.map(t => {
      if (t.id === lastRenderedTop) reachedOld = true;
      const fresh = !reachedOld && lastRenderedTop !== null;
      return `<div class="tx ${t.flagged ? 'flag' : ''} ${fresh ? 'fresh' : ''}" data-id="${esc(t.id)}" role="button" tabindex="0">
        <span class="c-time">${clock(t.received_at * 1000)}</span>
        <span class="c-user"><b>${esc(t.user_name)}</b><small>${esc(t.user_id)}</small></span>
        <span class="c-merch">${esc(t.merchant)}<small>${esc(t.category)}</small></span>
        <span class="c-loc">${esc(t.city)}, ${esc(t.country)}</span>
        <span class="c-amt">${inrFull(t.amount)}</span>
        <span class="c-risk risk" title="Risk score ${t.score.toFixed(2)}"><i style="width:${Math.round(t.score * 100)}%"></i><em>${t.score.toFixed(2)}</em></span>
        ${t.flagged ? `<p class="note">${esc(noteFor(t))}</p>` : ''}
      </div>`;
    }).join('');
    $('#feed').innerHTML = html;
    lastRenderedTop = list[0].id;
  }

  /* ------------------------------------------------------------ alerts view */
  function renderAlerts() {
    const list = alertOrder.map(id => alerts.get(id)).filter(a => a && (alertMode === 'all' || a.status === 'open')).slice(0, 50);
    if (!list.length) {
      $('#alerts').innerHTML = `<p class="empty">${alertMode === 'open' ? 'Inbox clear. New alerts appear here as they are flagged.' : 'No alerts yet.'}</p>`;
      return;
    }
    $('#alerts').innerHTML = list.map(a => {
      const done = a.status !== 'open';
      return `<article class="al ${done ? 'done' : ''}" data-id="${esc(a.id)}">
        <div class="al-top"><span class="al-amt">${inrFull(a.amount)}</span><span class="lvl ${a.level}">${a.level[0].toUpperCase() + a.level.slice(1)} risk, ${a.score.toFixed(2)}</span></div>
        <div class="al-who"><b>${esc(a.user_name)}</b> at ${esc(a.merchant)}, ${esc(a.city)}</div>
        <div class="al-why">${esc(noteFor(a))}</div>
        <div class="al-actions">
          ${done ? `<span class="status ${a.status}">${a.status === 'confirmed' ? 'Confirmed as fraud' : 'Marked safe'}</span>
                    <button class="btn ghost sm" data-act="open" data-id="${esc(a.id)}">Reopen</button>`
                 : `<button class="btn sm" data-act="confirmed" data-id="${esc(a.id)}">Confirm fraud</button>
                    <button class="btn ghost sm" data-act="dismissed" data-id="${esc(a.id)}">Mark safe</button>`}
          <button class="btn ghost sm" data-act="detail" data-id="${esc(a.id)}">Details</button>
        </div>
      </article>`;
    }).join('');
  }

  async function resolve(id, status) {
    try {
      await api(`/api/alerts/${encodeURIComponent(id)}/resolve`, 'POST', { status });
    } catch (e) { toast('Could not update alert', e.message, 'info'); }
  }

  /* ----------------------------------------------------------------- detail */
  function meterHtml(score, thr) {
    const bad = score >= thr;
    return `<div class="meter ${bad ? 'bad' : ''}"><i style="width:${Math.round(score * 100)}%"></i><u style="left:${thr * 100}%"></u></div>
            <div class="meter-cap">Risk score ${score.toFixed(2)}. Alert threshold ${thr.toFixed(2)} (black line).</div>`;
  }
  function reasonList(t) {
    if (!t.reasons.length) return '<ul class="why"><li><span>No risk signals found. This looks like normal behaviour for this customer.</span></li></ul>';
    return `<ul class="why">${t.reasons.map(r => `<li class="hit"><b>${esc(r.title)}</b> <span>${esc(r.detail)}</span></li>`).join('')}</ul>`;
  }
  function openDetail(id) {
    const t = alerts.get(id) || txns.find(x => x.id === id);
    if (!t) return;
    const d = $('#detail');
    const feats = (t.features || []).map(f => `<li><b>${esc(f.feature)}</b> <span>${f.z > 0 ? '+' : ''}${f.z} standard deviations from normal</span></li>`).join('');
    d.innerHTML = `<div class="dg">
      <div class="dg-h"><div><h2 id="d-title">${inrFull(t.amount)} at ${esc(t.merchant)}</h2>
        <p>${esc(t.id)}, ${new Date(t.received_at * 1000).toLocaleString()}</p></div>
        <button class="btn ghost sm" id="dClose" type="button">Close</button></div>
      ${meterHtml(t.score, config.threshold)}
      <dl class="dg-grid">
        <div><dt>Customer</dt><dd>${esc(t.user_name)} (${esc(t.user_id)})</dd></div>
        <div><dt>Location</dt><dd>${esc(t.city)}, ${esc(t.country)}</dd></div>
        <div><dt>Merchant type</dt><dd>${esc(t.category)}</dd></div>
        <div><dt>Device</dt><dd>${esc(t.device_id)}</dd></div>
        <div><dt>Model score</dt><dd>${t.ml_score.toFixed(2)}</dd></div>
        <div><dt>Rule score</dt><dd>${t.rule_score.toFixed(2)}</dd></div>
        <div><dt>Time to decide</dt><dd>${t.detection_ms ?? '-'} ms</dd></div>
        <div><dt>Status</dt><dd>${t.flagged ? ({ open: 'Needs review', confirmed: 'Confirmed fraud', dismissed: 'Marked safe' }[t.status]) : 'Approved'}</dd></div>
      </dl>
      <h3>Why this score</h3>${reasonList(t)}
      ${feats ? `<h3>Most unusual signals</h3><ul class="why">${feats}</ul>` : ''}
      ${t.flagged ? `<div class="dg-actions">
        <button class="btn" data-act="confirmed" data-id="${esc(t.id)}" type="button">Confirm fraud</button>
        <button class="btn ghost" data-act="dismissed" data-id="${esc(t.id)}" type="button">Mark safe</button></div>` : ''}
    </div>`;
    $('#dClose', d).onclick = () => d.close();
    d.showModal();
  }
  $('#detail').addEventListener('click', e => {
    const b = e.target.closest('[data-act]');
    if (b) { resolve(b.dataset.id, b.dataset.act).then(() => $('#detail').close()); }
    else if (e.target === $('#detail')) $('#detail').close();
  });

  /* ----------------------------------------------------------------- charts */
  const charts = {};
  const gridColor = () => hexA(cssVar('--muted') || '#888', 0.18);

  function initCharts() {
    if (typeof Chart === 'undefined') {
      $$('.chart').forEach(c => { c.innerHTML = '<p class="empty">Charts need internet access to load Chart.js.</p>'; });
      return;
    }
    Chart.defaults.font.family = '"Schibsted Grotesk", system-ui, sans-serif';
    Chart.defaults.font.size = 12;
    const common = { animation: false, responsive: true, maintainAspectRatio: false };

    charts.timeline = new Chart($('#chTimeline'), {
      data: { labels: [], datasets: [
        { type: 'line', label: 'All transactions', data: [], fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
        { type: 'bar', label: 'Flagged', data: [], borderRadius: 2, yAxisID: 'y1' },
      ] },
      options: { ...common, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 7, maxRotation: 0 } },
          y: { beginAtZero: true, title: { display: true, text: 'Transactions per 5 s' } },
          y1: { position: 'right', beginAtZero: true, grid: { display: false }, ticks: { precision: 0 }, title: { display: true, text: 'Flagged' } } } },
    });
    charts.reasons = new Chart($('#chReasons'), {
      type: 'doughnut',
      data: { labels: [], datasets: [{ data: [], borderWidth: 0 }] },
      options: { ...common, cutout: '62%', plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } } } },
    });
    charts.cats = new Chart($('#chCats'), {
      type: 'bar',
      data: { labels: [], datasets: [{ data: [], borderRadius: 3 }] },
      options: { ...common, indexAxis: 'y', plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true, ticks: { precision: 0 } }, y: { grid: { display: false } } } },
    });
    charts.hist = new Chart($('#chHist'), {
      type: 'bar',
      data: { labels: Array.from({ length: 20 }, (_, i) => (i * 0.05).toFixed(2)), datasets: [{ data: [], borderRadius: 2 }] },
      options: { ...common, plugins: { legend: { display: false }, tooltip: { callbacks: { title: i => `Score ${i[0].label} to ${(+i[0].label + 0.05).toFixed(2)}` } } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { type: 'logarithmic', min: 1, title: { display: true, text: 'Transactions' } } } },
    });
    restyleCharts();
  }

  function restyleCharts() {
    if (!charts.timeline) return;
    const accent = cssVar('--accent'), alert = cssVar('--alert'), muted = cssVar('--muted');
    Chart.defaults.color = muted;
    Chart.defaults.borderColor = gridColor();
    const t = charts.timeline.data.datasets;
    t[0].borderColor = accent; t[0].backgroundColor = hexA(accent, 0.12);
    t[1].backgroundColor = alert;
    charts.cats.data.datasets[0].backgroundColor = accent;
    for (const c of Object.values(charts)) {
      for (const sc of Object.values(c.options.scales || {})) {
        sc.grid = { ...(sc.grid || {}), color: gridColor() };
        if (sc.title) sc.title.color = muted;
      }
      c.update('none');
    }
    if (stats) updateCharts(stats);
  }

  function updateCharts(s) {
    if (!charts.timeline) return;
    const tl = s.timeline;
    charts.timeline.data.labels = tl.total.map((_, i) => clock((tl.end - (tl.total.length - 1 - i) * tl.step) * 1000));
    charts.timeline.data.datasets[0].data = tl.total;
    charts.timeline.data.datasets[1].data = tl.flagged;
    charts.timeline.update('none');

    const rs = Object.entries(s.reasons).sort((a, b) => b[1] - a[1]);
    charts.reasons.data.labels = rs.map(r => REASON_NAMES[r[0]] || r[0]);
    charts.reasons.data.datasets[0].data = rs.map(r => r[1]);
    charts.reasons.data.datasets[0].backgroundColor = rs.map((_, i) => PALETTE[i % PALETTE.length]);
    charts.reasons.update('none');

    const cs = Object.entries(s.categories);
    charts.cats.data.labels = cs.map(c => c[0]);
    charts.cats.data.datasets[0].data = cs.map(c => c[1]);
    charts.cats.update('none');

    const accent = cssVar('--accent'), alert = cssVar('--alert');
    charts.hist.data.datasets[0].data = s.score_hist.map(v => v || null);
    charts.hist.data.datasets[0].backgroundColor = s.score_hist.map((_, i) => i * 0.05 >= s.threshold - 1e-9 ? alert : accent);
    charts.hist.update('none');
  }

  /* -------------------------------------------------------------------- map */
  let map = null, tiles = null;
  const okMarks = [], badMarks = [];
  function initMap() {
    if (typeof L === 'undefined') { $('#map').innerHTML = '<p class="empty">The map needs internet access to load Leaflet and map tiles.</p>'; return; }
    map = L.map('map', { worldCopyJump: true, minZoom: 2 }).setView([22, 45], 2);
    setTiles();
    setTimeout(() => map.invalidateSize(), 300);
  }
  function setTiles() {
  if (!map) return;
  if (tiles) map.removeLayer(tiles);
  const dark = document.documentElement.dataset.theme === 'dark';
  tiles = L.tileLayer(`https://{s}.basemaps.cartocdn.com/${dark ? 'dark_all' : 'light_all'}/{z}/{x}/{y}{r}.png`, {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 10,
  }).addTo(map);
}
  
  function plot(t) {
    if (!map) return;
    if (t.flagged) {
      const m = L.marker([t.lat, t.lon], { icon: L.divIcon({ className: '', html: '<span class="fdot"></span>', iconSize: [14, 14] }) })
        .addTo(map).bindTooltip(`${inrFull(t.amount)} at ${t.merchant}, ${t.city}`);
      badMarks.push(m);
      if (badMarks.length > 60) map.removeLayer(badMarks.shift());
    } else {
      const m = L.circleMarker([t.lat, t.lon], { radius: 3, weight: 0, fillColor: cssVar('--accent'), fillOpacity: 0.45 }).addTo(map);
      okMarks.push(m);
      if (okMarks.length > 120) map.removeLayer(okMarks.shift());
    }
  }
  function clearMap() { [...okMarks, ...badMarks].forEach(m => map && map.removeLayer(m)); okMarks.length = badMarks.length = 0; }

  /* ---------------------------------------------------------- socket events */
  function onTxns(list) {
    let newFlag = null;
    for (const t of list) {
      txns.unshift(t);
      plot(t);
      if (t.flagged) {
        alerts.set(t.id, t); alertOrder.unshift(t.id);
        if (!newFlag || t.score > newFlag.score) newFlag = t;
      }
    }
    if (txns.length > 150) txns.length = 150;
    if (alertOrder.length > 200) alertOrder.length = 200;
    return newFlag;
  }

  function onMessage(m) {
    switch (m.type) {
      case 'snapshot':
        txns = []; alerts.clear(); alertOrder = [];
        m.alerts.forEach(a => { alerts.set(a.id, a); });
        alertOrder = m.alerts.map(a => a.id).reverse();
        txns = m.txns.slice().reverse();
        clearMap(); m.txns.forEach(plot);
        applyConfig(m.config); applyStats(m.stats);
        lastRenderedTop = null; renderFeed(); renderAlerts();
        break;
      case 'tick': {
        const flagged = onTxns(m.txns);
        applyConfig(m.config); applyStats(m.stats);
        if (m.txns.length) renderFeed();
        if (m.txns.some(t => t.flagged)) renderAlerts();
        if (flagged && flagged.level === 'critical' && Date.now() - lastToast > 2500) {
          lastToast = Date.now();
          toast('Critical alert', `${inrFull(flagged.amount)} at ${flagged.merchant}, ${flagged.city}. ${noteFor(flagged)}`, '', () => openDetail(flagged.id));
          if (soundOn) beep();
        }
        break;
      }
      case 'alert_update': {
        const a = alerts.get(m.id); if (a) a.status = m.status;
        const t = txns.find(x => x.id === m.id); if (t) t.status = m.status;
        applyStats(m.stats); renderAlerts();
        break;
      }
      case 'config': applyConfig(m.config); break;
      case 'reset':
        txns = []; alerts.clear(); alertOrder = []; clearMap(); lastRenderedTop = null;
        applyConfig(m.config); applyStats(m.stats); renderFeed(); renderAlerts();
        break;
    }
  }

  let ws = null, retry = 0;
  function connect() {
    setConn('connecting');
    ws = new WebSocket(WS_URL);
    ws.onopen = () => { retry = 0; setConn(config.paused ? 'paused' : 'live'); };
    ws.onmessage = e => { try { onMessage(JSON.parse(e.data)); } catch (err) { console.error(err); } };
    ws.onclose = () => { setConn('down'); setTimeout(connect, Math.min(5000, 800 * ++retry)); };
    ws.onerror = () => ws.close();
  }
  setInterval(() => { if (ws && ws.readyState === 1) ws.send('ping'); }, 20000);

  /* ------------------------------------------------------------- UI wiring */
  $$('[data-feed]').forEach(b => b.onclick = () => {
    feedMode = b.dataset.feed;
    $$('[data-feed]').forEach(x => { const on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-selected', on); });
    lastRenderedTop = null; renderFeed();
  });
  $$('[data-alerts]').forEach(b => b.onclick = () => {
    alertMode = b.dataset.alerts;
    $$('[data-alerts]').forEach(x => { const on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-selected', on); });
    renderAlerts();
  });
  $('#feedSearch').addEventListener('input', e => { query = e.target.value.trim().toLowerCase(); lastRenderedTop = null; renderFeed(); });
  $('#feed').addEventListener('click', e => { const r = e.target.closest('.tx'); if (r) openDetail(r.dataset.id); });
  $('#feed').addEventListener('keydown', e => { if (e.key === 'Enter') { const r = e.target.closest('.tx'); if (r) openDetail(r.dataset.id); } });
  $('#alerts').addEventListener('click', e => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    if (b.dataset.act === 'detail') openDetail(b.dataset.id); else resolve(b.dataset.id, b.dataset.act);
  });

  $('#btnPause').onclick = () => api('/api/config', 'POST', { paused: !config.paused }).catch(e => toast('Could not change the stream', e.message, 'info'));
  const slider = (id, key, conv) => {
    const el = $('#' + id);
    el.addEventListener('input', () => {
      const v = +el.value;
      $('#v-' + { rate: 'rate', thr: 'thr', fr: 'fr' }[id]).textContent = id === 'thr' ? v.toFixed(2) : id === 'fr' ? v + '%' : v;
    });
    el.addEventListener('change', () => api('/api/config', 'POST', { [key]: conv(+el.value) }).catch(e => toast('Could not save setting', e.message, 'info')));
  };
  slider('rate', 'rate', v => v); slider('thr', 'threshold', v => v); slider('fr', 'fraud_rate', v => v / 100);

  $$('[data-attack]').forEach(b => b.onclick = async () => {
    try {
      const r = await api('/api/simulate/attack?kind=' + b.dataset.attack, 'POST');
      toast('Attack launched', `${r.queued} suspicious payments are on their way into the stream.`, 'info');
    } catch (e) { toast('Could not launch attack', e.message, 'info'); }
  });
  $('#btnReset').onclick = () => api('/api/reset', 'POST').then(() => toast('Counters reset', 'Customer history is kept so detection stays accurate.', 'info'));
  $('#btnExport').href = API + '/api/export/alerts.csv';

  /* ----------------------------------------------------------- risk checker */
  const PRESETS = {
    normal: { amount: 650, category: 'Groceries', city: null, dev: 0, min: 240, vel: 0 },
    abroad: { amount: 85000, category: 'Electronics', city: 'London', dev: 1, min: 6, vel: 0 },
    burst: { amount: 2400, category: 'Gift Cards', city: null, dev: 0, min: 1, vel: 5 },
    crypto: { amount: 120000, category: 'Crypto Exchange', city: 'Mumbai', dev: 1, min: 45, vel: 0 },
  };
  async function initChecker() {
    try {
      const meta = await api('/api/meta');
      $('#c-cat').innerHTML = meta.categories.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join('');
      $('#c-city').innerHTML = meta.cities.map(c => `<option value="${esc(c.name)}">${esc(c.name)} (${esc(c.country)})</option>`).join('');
      $('#c-cat').value = 'Electronics'; $('#c-city').value = 'Mumbai';
    } catch { /* server not ready yet; retry once */ setTimeout(initChecker, 2000); }
  }
  $$('[data-preset]').forEach(b => b.onclick = () => {
    const p = PRESETS[b.dataset.preset];
    $('#c-amt').value = p.amount; $('#c-cat').value = p.category;
    if (p.city) $('#c-city').value = p.city;
    $('#c-dev').value = p.dev; $('#c-min').value = p.min; $('#c-vel').value = p.vel;
    $('#chkForm').requestSubmit();
  });
  $('#chkForm').addEventListener('submit', async e => {
    e.preventDefault();
    const out = $('#chkOut');
    try {
      const r = await api('/api/score', 'POST', {
        amount: +$('#c-amt').value, category: $('#c-cat').value, city: $('#c-city').value,
        new_device: $('#c-dev').value === '1', minutes_since_last: +$('#c-min').value, txns_last_minute: +$('#c-vel').value,
      });
      const thr = config.threshold, bad = r.score >= thr;
      out.innerHTML = `<div class="verdict ${bad ? 'bad' : 'good'}"><strong>${Math.round(r.score * 100)}</strong>
        <span class="v-text">${bad ? 'Would be flagged for review' : 'Would be approved'}</span>
        <span class="muted sm">decided in ${r.detection_ms} ms</span></div>
        ${meterHtml(r.score, thr)}${reasonList(r)}`;
    } catch (err) {
      out.innerHTML = `<p class="empty">Could not check this payment: ${esc(err.message)}</p>`;
    }
  });

  /* ------------------------------------------------------------------- boot */
document.documentElement.dataset.theme = 'dark';
if ($('#btnTheme')) $('#btnTheme').textContent = 'Light mode';
initCharts();
initMap();
initChecker();
connect();
})();