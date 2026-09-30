import { useEffect, useRef, useState } from "react";
import { fetchFrame } from "../lib/api.js";
import { drawProbability } from "../lib/canvas.js";

export default function Hero({ config }) {
  const canvasRef = useRef(null);
  const [prob, setProb] = useState(null);
  const lead = 60;

  useEffect(() => {
    let cancelled = false;
    fetchFrame(0, lead, 64).then((data) => {
      if (cancelled) return;
      setProb(data);
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        drawProbability(ctx, canvas.width, canvas.height, data.probability, data.strikes);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const peakPct = prob ? Math.round(prob.maxProb * Math.max(0, 1 - lead / 220) * 100) : null;

  return (
    <section className="hero" id="dashboard">
      <div className="eyebrow">Thunderstorm &amp; lightning nowcasting</div>
      <h1>
        See the storm
        <br />
        before it forms.
      </h1>
      <p className="lead">
        Vidyut fuses satellite, radar, lightning and atmospheric data into one model that tells you
        where convection is building — up to three hours ahead.
      </p>
      <div className="ctas">
        <a className="btn btn-primary" href="#try">
          Try live sample
        </a>
        <a className="btn btn-outline" href="#how">
          How it works
        </a>
      </div>

      {config?.offline && (
        <p className="offline-note">
          Running in offline demo mode — start <code>src/api.py</code> to see the real pipeline.
        </p>
      )}

      <div className="device-wrap" id="try">
        <div className="device-blob" />
        <div className="device">
          <div className="device-cam" />
          <div className="screen">
            <div className="screen-top">
              <div className="path">
                <span>Nowcast</span> &nbsp;›&nbsp; Live
              </div>
              <div className="region-pill">{config?.region || "Punjab Region"} ▾</div>
            </div>
            <div className="stat-row">
              <div className="stat-num">{peakPct !== null ? `${peakPct}%` : "—%"}</div>
              <div className="stat-label">Peak storm probability, next {lead} min</div>
            </div>
            <div className="screen-canvas-wrap">
              <canvas ref={canvasRef} className="radarcanvas" width={860} height={360} />
              <div className="strike-legend">× lightning strike, last window</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
