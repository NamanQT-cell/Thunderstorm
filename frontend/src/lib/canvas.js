export function drawSatellite(ctx, w, h, sat) {
  const n = sat.length,
    cw = w / n,
    ch = h / n;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const v = sat[y][x];
      const g = Math.round(230 - v * 180);
      ctx.fillStyle = `rgb(${g},${g + 6},${g + 2})`;
      ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
    }
  }
}

const HEAT_STOPS = [
  [255, 247, 213],
  [250, 196, 110],
  [224, 111, 50],
  [176, 42, 42],
];

export function heatColor(v) {
  const t = Math.min(0.999, v) * (HEAT_STOPS.length - 1);
  const i = Math.floor(t),
    f = t - i;
  const a = HEAT_STOPS[i],
    b = HEAT_STOPS[Math.min(i + 1, HEAT_STOPS.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

export function drawProbability(ctx, w, h, prob, strikes) {
  const n = prob.length,
    cw = w / n,
    ch = h / n;
  ctx.fillStyle = "#151b16";
  ctx.fillRect(0, 0, w, h);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      ctx.fillStyle = heatColor(prob[y][x]);
      ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
    }
  }
  ctx.strokeStyle = "#4f7cff";
  ctx.lineWidth = 2;
  strikes.forEach(([x, y]) => {
    const cx = (x + 0.5) * cw,
      cy = (y + 0.5) * ch,
      s = Math.max(3, cw * 0.5);
    ctx.beginPath();
    ctx.moveTo(cx - s, cy - s);
    ctx.lineTo(cx + s, cy + s);
    ctx.moveTo(cx + s, cy - s);
    ctx.lineTo(cx - s, cy + s);
    ctx.stroke();
  });
}

export function alertFor(maxProb) {
  if (maxProb > 0.75) return { label: "Severe", color: "#b0392f" };
  if (maxProb > 0.5) return { label: "High", color: "#c96a2e" };
  if (maxProb > 0.3) return { label: "Moderate", color: "#b8912c" };
  return { label: "Low", color: "#4a7c3f" };
}
