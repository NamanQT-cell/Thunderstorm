const SOURCES = [
  "INSAT-3D / 3DR satellite",
  "IMD radar",
  "GPM IMERG rainfall",
  "ERA5 atmospheric fields",
  "Lightning strike network",
];

export default function Sources() {
  return (
    <section className="sources" id="sources">
      <div className="lbl">Every forecast is fused from</div>
      <div className="source-row">
        {SOURCES.map((s) => (
          <span className="source-chip" key={s}>
            <i />
            {s}
          </span>
        ))}
      </div>
    </section>
  );
}
