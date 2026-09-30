import { useEffect, useRef, useState } from "react";
import { fetchFrame } from "../lib/api.js";
import { drawSatellite, drawProbability, alertFor } from "../lib/canvas.js";

export default function DashboardSplit() {
  const canvasRef = useRef(null);
  const [timeStep, setTimeStep] = useState(0);
  const [lead, setLead] = useState(60);
  const [grid, setGrid] = useState(64);
  const [frame, setFrame] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchFrame(timeStep, lead, grid).then((data) => {
      if (cancelled) return;
      setFrame(data);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [timeStep, lead, grid]);

  useEffect(() => {
    if (!frame) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const w = canvas.width,
      h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    drawSatellite(ctx, w, h * 0.48, frame.satellite);
    ctx.save();
    ctx.translate(0, h * 0.52);
    drawProbability(ctx, w, h * 0.48, frame.probability, frame.strikes);
    ctx.restore();
    ctx.fillStyle = "rgba(255,255,255,.85)";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText("Observed (satellite)", 8, 16);
    ctx.fillStyle = "rgba(255,255,255,.9)";
    ctx.fillText(`Predicted probability +${lead}m`, 8, h * 0.52 + 16);
  }, [frame, lead]);

  const decayedMax = frame ? frame.maxProb * Math.max(0, 1 - lead / 220) : 0;
  const alert = frame ? alertFor(decayedMax) : null;

  return (
    <section className="split" id="how">
      <div className="visual">
        <canvas ref={canvasRef} width={420} height={420} />
      </div>
      <div>
        <div className="eyebrow">See the full picture</div>
        <h2>Move through the last three hours.</h2>
        <p style={{ color: "var(--muted)" }}>
          Drag the controls below — they redraw the observed field, the predicted probability and
          the alert level exactly the way <code>src/api.py</code> computes them from the real
          pipeline, frame by frame.
        </p>
        <ul className="bullets">
          <li>
            <span className="num">01</span>Slide back in time to see how convection built up before
            the current moment.
          </li>
          <li>
            <span className="num">02</span>Change the forecast lead time to see confidence fall off
            with distance.
          </li>
          <li>
            <span className="num">03</span>Raise the grid resolution for a sharper — but noisier —
            view of each cell.
          </li>
        </ul>

        <div className="controls">
          <label>
            Time step (minutes from now)
            <input
              type="range"
              min={-180}
              max={0}
              step={30}
              value={timeStep}
              onChange={(e) => setTimeStep(parseInt(e.target.value, 10))}
            />
          </label>
          <div className="row">
            <label style={{ flex: 1 }}>
              Lead time
              <select value={lead} onChange={(e) => setLead(parseInt(e.target.value, 10))}>
                {[30, 60, 90, 120, 150, 180].map((lt) => (
                  <option key={lt} value={lt}>
                    +{lt} min
                  </option>
                ))}
              </select>
            </label>
            <label style={{ flex: 1 }}>
              Grid resolution
              <select value={grid} onChange={(e) => setGrid(parseInt(e.target.value, 10))}>
                {[32, 64, 96].map((g) => (
                  <option key={g} value={g}>
                    {g} × {g}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div
            className="alert-badge"
            style={{ borderLeft: `4px solid ${alert ? alert.color : "#ccc"}` }}
          >
            {loading || !frame ? (
              "Loading…"
            ) : (
              <>
                Alert level: <strong>{alert.label}</strong> · peak{" "}
                {Math.round(decayedMax * 100)}% · mean {Math.round(frame.meanProb * 100)}%
                {frame.usedRealModel ? " · trained model" : " · demo formula"}
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
