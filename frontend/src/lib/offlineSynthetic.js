// Pure client-side stand-in for src/api.py, used only when the real API
// isn't reachable. It never talks to the model -- it just gives the UI
// something believable to render so the page still works standalone.

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCells(rng, n) {
  const cells = [];
  for (let i = 0; i < n; i++) {
    cells.push({ x: rng(), y: rng(), r: 0.12 + rng() * 0.22, strength: 0.5 + rng() * 0.7 });
  }
  return cells;
}

function fieldAt(gx, gy, cells) {
  let v = 0;
  for (const c of cells) {
    const dx = gx - c.x,
      dy = gy - c.y;
    const d2 = (dx * dx + dy * dy) / (c.r * c.r);
    v += c.strength * Math.exp(-d2);
  }
  return Math.max(0, Math.min(1, v));
}

export function computeOfflineFrame(timeStep, lead, grid) {
  const seed = Math.abs(timeStep) * 7 + grid * 13 + 1;
  const rng = mulberry32(seed);
  const cells = makeCells(rng, 4 + Math.floor(rng() * 3));
  const satellite = [],
    probability = [],
    strikes = [];

  for (let y = 0; y < grid; y++) {
    const rowSat = [],
      rowProb = [];
    for (let x = 0; x < grid; x++) {
      const gx = x / grid,
        gy = y / grid;
      const s = fieldAt(gx, gy, cells);
      const noise = (rng() - 0.5) * 0.08;
      const p = Math.max(0, Math.min(1, s * 0.85 + noise));
      rowSat.push(s);
      rowProb.push(p);
      if (p > 0.55 && rng() < 0.14) strikes.push([x, y]);
    }
    satellite.push(rowSat);
    probability.push(rowProb);
  }

  let max = 0,
    sum = 0;
  for (const row of probability) for (const v of row) { if (v > max) max = v; sum += v; }
  const mean = sum / (grid * grid);

  const leadTimeProbs = {};
  [30, 60, 90].forEach((lt) => {
    leadTimeProbs[String(lt)] = Math.round(Math.max(0, Math.min(1, max * (1 - lt / 200))) * 1000) / 10;
  });

  return {
    usedRealModel: false,
    satellite,
    probability,
    strikes,
    maxProb: max,
    meanProb: mean,
    totalStrikes: strikes.length,
    leadTimeProbs,
  };
}

export function computeOfflineTrend(grid) {
  const minutes = [];
  const maxProbability = [];
  for (let m = -180; m <= 30; m += 30) {
    minutes.push(m);
    const { maxProb } = computeOfflineFrame(m, 60, grid);
    maxProbability.push(maxProb);
  }
  return { minutes, maxProbability };
}
