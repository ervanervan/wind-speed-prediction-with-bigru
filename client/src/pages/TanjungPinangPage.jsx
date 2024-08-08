import React from "react";

import FF_X_TPI from "./FF_X_TPI";
import FF_AVG_TPI from "./FF_AVG_TPI";

import Tabs from "../components/Tabs";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

import PatternHero from "../assets/pattern_hero.svg";
import gradientTL from "../assets/gradient-t-l.svg";
import gradientBL from "../assets/gradient-b-l.svg";
import gradientTR from "../assets/gradient-t-r.svg";
import gradientBR from "../assets/gradient-b-r.svg";

import IconAir from "../assets/icons/IconAir";

const TanjungPinangPage = () => {
  const tabs = [
    {
      label: "Ff_Avg",
      icon: IconAir,
      content: <FF_AVG_TPI />,
    },
    {
      label: "Ff_X",
      icon: IconAir,
      content: <FF_X_TPI />,
    },
  ];

  return (
    <>
      <section className="relative">
        <Navbar />
        <img src={gradientTL} alt="" className="absolute top-12 left-0 -z-10" />
        <img src={gradientTR} alt="" className="absolute top-0 right-0 -z-10" />
        <img
          src={gradientBL}
          alt=""
          className="absolute -bottom-40 left-0 -z-10"
        />
        <img
          src={gradientBR}
          alt=""
          className="absolute bottom-0 right-0 -z-10"
        />

        <div className="container mx-auto relative">
          <div>
            <div className="md:my-12 px-5 py-20 relative">
              <img
                src={PatternHero}
                alt=""
                className="absolute top-1/2 -translate-y-1/2 left-1/2 transform -translate-x-1/2 -z-10 hidden lg:block"
              />
              <h1 className="text-white-2 font-medium text-4xl lg:text-5xl flex items-center justify-center text-center leading-[3.5rem] lg:leading-[4rem]">
                Prediksi Kecepatan Angin <br />
                Kota Tanjungpinang
              </h1>
            </div>
          </div>
        </div>
      </section>

      <Tabs tabs={tabs} />
      <Footer />
    </>
  );
};

export default TanjungPinangPage;
