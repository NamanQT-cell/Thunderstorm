export default function Testimonial() {
  return (
    <section className="testi">
      <div className="art">
        <div className="ring" style={{ width: "70%", height: "70%", top: "15%", left: "15%" }} />
        <div className="ring" style={{ width: "46%", height: "46%", top: "27%", left: "27%" }} />
        <div className="ring" style={{ width: "22%", height: "22%", top: "39%", left: "39%" }} />
      </div>
      <div>
        <blockquote>
          "We moved the evening market indoors forty minutes before the first strike hit. That
          warning came from the probability map, not the sky."
        </blockquote>
        <div className="who">District disaster management coordinator, early pilot site</div>
      </div>
    </section>
  );
}
