import React from "react";
import PatternHero from "../assets/pattern_hero.svg";
import gradientTL from "../assets/gradient-t-l.svg";
import gradientBL from "../assets/gradient-b-l.svg";
import gradientTR from "../assets/gradient-t-r.svg";
import gradientBR from "../assets/gradient-b-r.svg";
import Navbar from "../components/Navbar";
import Tabs from "../components/Tabs";

const AnambasPage = () => {
  const tabs = [
    {
      label: "Ff_Avg",
      // icon: HomeIcon,
      content: <div>Content for Tab 1</div>,
    },
    {
      label: "Ff_x",
      // icon: HomeIcon,
      content: <div>Content for Tab 2</div>,
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
            <div className="md:mt-12 px-5 py-20 relative">
              <img
                src={PatternHero}
                alt=""
                className="absolute top-1/2 -translate-y-1/2 left-1/2 transform -translate-x-1/2 -z-10 hidden lg:block"
              />
              <h1 className="text-white-2 font-medium text-4xl lg:text-5xl flex items-center justify-center text-center leading-[3.5rem] md:leading-[4rem]">
                Prediksi Kecepatan Angin <br />
                Kabupaten Kepulauan Anambas
              </h1>
            </div>
          </div>
        </div>
      </section>

      <Tabs tabs={tabs} />
    </>
  );
};

export default AnambasPage;
