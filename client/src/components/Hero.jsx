import React from "react";
import PatternImage from "../assets/Pattern-Image.png";
import PatternHero from "../assets/pattern_hero.svg";
import gradientTL from "../assets/gradient-t-l.svg";
import gradientBL from "../assets/gradient-b-l.svg";
import gradientTR from "../assets/gradient-t-r.svg";
import gradientBR from "../assets/gradient-b-r.svg";
import Navbar from "./Navbar";

export default function Hero() {
  return (
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
        <div className="flex flex-col md:gap-20">
          <div className="md:mt-12 px-5 py-20 relative">
            <img
              src={PatternHero}
              alt=""
              className="absolute top-1/2 -translate-y-1/2 left-1/2 transform -translate-x-1/2 -z-10 hidden lg:block"
            />
            <h1 className="text-white-2 font-medium text-4xl md:text-5xl flex items-center justify-center text-center leading-[3.5rem] md:leading-[4rem]">
              Implementasi algoritma <br /> Bidirectional Gated Recurrent Unit{" "}
              <br />
              (BiGRU) untuk Prediksi Kecepatan Angin
            </h1>
          </div>
          <div className="flex items-center justify-center w-full px-5 py-5">
            <img src={PatternImage} alt="image hero" />
          </div>
        </div>
      </div>
    </section>
  );
}
