const STEPS = [
  {
    n: "01",
    title: "Ingest",
    body: "Satellite, radar, lightning and NWP loaders pull the latest frames and normalize them onto one grid.",
  },
  {
    n: "02",
    title: "Fuse & predict",
    body: "A CNN trained on lightning-derived labels turns the merged frame into a per-cell probability map.",
  },
  {
    n: "03",
    title: "Alert",
    body: "Probabilities cross a threshold, the region gets a severity level, and the trend is tracked over the following hours.",
  },
];

export default function Steps() {
  return (
    <section className="steps">
      <div className="steps-top">
        <h2>From raw data to an alert</h2>
        <a className="btn btn-pale" href="#dashboard">
          Open the dashboard
        </a>
      </div>
      <div className="stepgrid">
        {STEPS.map((s) => (
          <div key={s.n}>
            <div className="n">{s.n}</div>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
