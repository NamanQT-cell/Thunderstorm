import { useEffect, useRef } from "react";
import { heatColor } from "../lib/canvas.js";

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function Banner({ caption, seed = 7 }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    const ctx = canvas.getContext("2d");
    let raf;
    let angle = 0;

    const rng = mulberry32(seed);
    const blobs = [];
    for (let i = 0; i < 10; i++) {
      blobs.push({ x: rng(), y: rng(), r: 0.15 + rng() * 0.35, s: 0.3 + rng() * 0.6 });
    }

    function size() {
      canvas.width = wrap.clientWidth;
      canvas.height = wrap.clientHeight;
    }
    size();
    window.addEventListener("resize", size);

    function frame() {
      const w = canvas.width,
        h = canvas.height;
      ctx.fillStyle = "#12180f";
      ctx.fillRect(0, 0, w, h);

      for (let y = 0; y < h; y += 6) {
        for (let x = 0; x < w; x += 6) {
          let val = 0;
          for (const b of blobs) {
            const dx = x / w - b.x,
              dy = y / h - b.y;
            val += b.s * Math.exp(-(dx * dx + dy * dy) / (b.r * b.r));
          }
          val = Math.min(1, val);
          if (val > 0.08) {
            ctx.fillStyle = heatColor(val);
            ctx.globalAlpha = 0.85;
            ctx.fillRect(x, y, 6, 6);
          }
        }
      }
      ctx.globalAlpha = 1;

      const cx = w / 2,
        cy = h / 2,
        R = Math.max(w, h);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      const grad = ctx.createLinearGradient(0, 0, R, 0);
      grad.addColorStop(0, "rgba(142,156,120,0.55)");
      grad.addColorStop(1, "rgba(142,156,120,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, R, -0.18, 0.18);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      angle += 0.008;
      raf = requestAnimationFrame(frame);
    }
    frame();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", size);
    };
  }, [seed]);

  return (
    <div className="banner" ref={wrapRef}>
      <canvas ref={canvasRef} />
      <div className="cap">{caption}</div>
    </div>
  );
}
