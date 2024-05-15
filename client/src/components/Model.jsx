import React from "react";
import model from "../assets/model.svg";
import Location from "./Location";

export default function Model() {
  return (
    <section className="relative">
      <div className="container mx-auto">
        <div className="flex md:pt-20 md:pb-12 gap-x-32 items-start px-5 py-8 md:px-0">
          <img src={model} alt="" className="hidden md:block" />
          <div className="flex flex-col gap-6">
            <h1 className="text-3xl md:text-4xl font-medium text-white-2">
              Bidirectional Gated Recurrent Unit (BiGRU)
            </h1>
            <p className="text-lg text-white-3 xl:w-[69%]">
              Merupakan pengembangan dari konsep GRU yang memperkenalkan
              pemrosesan inputan dalam dua arah yaitu maju dan mundur secara
              bersamaan dengan menggunakan lapisan GRU terpisah yang
              memungkinkan untuk menangkap informasi dari konteks masa lalu dan
              masa depan.
            </p>
          </div>
        </div>
      </div>
      <Location />
    </section>
  );
}
