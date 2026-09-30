const FEATURES = [
  {
    title: "Multi-source fusion",
    body: "Brightness temperature, reflectivity, lightning density and CAPE/shear are stacked into one input frame per time step.",
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#14150e" strokeWidth="1.4">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 3a9 9 0 0 1 0 18" />
      </svg>
    ),
  },
  {
    title: "Lightning-informed labels",
    body: "Historical strikes become the ground truth the model is trained against, not a guess bolted on afterwards.",
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#14150e" strokeWidth="1.4">
        <path d="M13 2 4 14h6l-1 8 9-12h-6z" />
      </svg>
    ),
  },
  {
    title: "Cell-level probability",
    body: "Output is a per-cell heatmap, not a single regional number, so you see exactly where risk is concentrated.",
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#14150e" strokeWidth="1.4">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    title: "Lead-time forecasts",
    body: "Choose 30 to 180 minutes ahead and watch confidence shift as fresher data comes in.",
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#14150e" strokeWidth="1.4">
        <path d="M4 19h16M7 15l4-6 3 4 4-7" />
      </svg>
    ),
  },
];

export default function Features() {
  return (
    <section className="features">
      <div className="eyebrow" style={{ textAlign: "center" }}>
        What the model sees
      </div>
      <h2>One grid, four kinds of evidence.</h2>
      <p>Each source tells a different part of the story. Vidyut merges them into a single array before a single prediction is made.</p>
      <div className="fgrid">
        {FEATURES.map((f) => (
          <div key={f.title}>
            {f.icon}
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
