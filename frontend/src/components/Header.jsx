import { useEffect, useState } from "react";

export default function Header() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={scrolled ? "scrolled" : ""}>
      <div className="wrap nav">
        <div className="brand">
          <span className="mark" />
          Vidyut
        </div>
        <nav className="navlinks">
          <a href="#dashboard">Live Dashboard</a>
          <a href="#how">How it Works</a>
          <a href="#sources">Data Sources</a>
          <a href="#compare">Why Vidyut</a>
        </nav>
        <a className="btn btn-primary" href="#dashboard">
          Open Dashboard ↗
        </a>
      </div>
    </header>
  );
}
