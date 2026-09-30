const ROWS = [
  ["Works without dedicated radar", "yes", "Requires radar coverage", "no", "No hardware dependency", "yes"],
  ["Sub-hourly probability updates", "yes", "Near real-time", "yes", "Hours between updates", "no"],
  ["Lightning-trained labels", "yes", "Reflectivity only", "no", "Forecaster judgment", "no"],
  ["Lead time up to 3 hours", "yes", "Typically <30 min", "no", "Variable, expert-dependent", "yes"],
  ["Runs in open-data mode", "yes", "Needs institutional access", "no", "No data dependency", "yes"],
];

export default function Compare() {
  return (
    <section className="why" id="compare">
      <div className="eyebrow">Why Vidyut?</div>
      <h2>Built for places radar alone can't reach.</h2>
      <p>
        Most of the country sits outside dense Doppler coverage. Vidyut is designed to keep working
        even where radar access is slow or absent.
      </p>
      <a className="btn btn-pale" href="#dashboard">
        See it in action
      </a>
      <table className="cmp">
        <tbody>
          <tr>
            <th>Vidyut</th>
            <th>Radar-only nowcasting</th>
            <th>Manual forecasting</th>
          </tr>
          {ROWS.map(([a, aType, b, bType, c, cType], i) => (
            <tr key={i}>
              <td className={aType}>{a}</td>
              <td className={bType}>{b}</td>
              <td className={cType}>{c}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
