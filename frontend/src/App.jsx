import { useEffect, useState } from "react";
import { fetchConfig } from "./lib/api.js";

import Header from "./components/Header.jsx";
import Hero from "./components/Hero.jsx";
import Sources from "./components/Sources.jsx";
import Features from "./components/Features.jsx";
import Banner from "./components/Banner.jsx";
import DashboardSplit from "./components/DashboardSplit.jsx";
import Compare from "./components/Compare.jsx";
import Testimonial from "./components/Testimonial.jsx";
import Steps from "./components/Steps.jsx";
import CTA from "./components/CTA.jsx";
import Footer from "./components/Footer.jsx";

export default function App() {
  const [config, setConfig] = useState(null);

  useEffect(() => {
    fetchConfig().then(setConfig);
  }, []);

  return (
    <>
      <Header />
      <main className="wrap">
        <Hero config={config} />
        <Sources />
        <Features />
        <Banner caption="Live radar composite — synthetic demo mode" seed={7} />
        <DashboardSplit />
        <Compare />
        <Testimonial />
        <Steps />
        <Banner caption="Composite view — satellite + radar + lightning, layered" seed={19} />
        <CTA />
      </main>
      <Footer />
    </>
  );
}
