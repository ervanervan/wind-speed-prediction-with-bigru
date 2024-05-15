import React from "react";
import location from "../assets/lokasi.svg";
import img1 from "../assets/unsplash_DuaR5Shk3E8.png";
import img2 from "../assets/unsplash_OZdtnIeK_uY.png";
import img3 from "../assets/demom0-2s4xmYSOgjc-unsplash 1.png";
import img4 from "../assets/deepavali-gaind-f0BHMYrG0fc-unsplash 1.png";
import gradientLTR from "../assets/gradient-l-t-r.svg";
import gradientLCL from "../assets/gradient-l-c-l.svg";
import gradientLBR from "../assets/gradient-l-b-r.svg";

export default function Location() {
  return (
    <section className="relative">
      <img
        src={gradientLTR}
        alt=""
        className="absolute -top-48 right-0 -z-10"
      />
      <img src={gradientLCL} alt="" className="absolute top-0 left-0 -z-10" />
      <img
        src={gradientLBR}
        alt=""
        className="absolute bottom-0 right-0 -z-10"
      />
      <div className="container mx-auto">
        <div className="flex flex-col lg:flex-row gap-y-8 py-12 xl:gap-x-32 items-start px-5">
          <img src={location} alt="" className="hidden md:block" />
          <div className="flex flex-col gap-16">
            <div className="flex flex-col gap-6">
              <h1 className="text-3xl md:text-4xl font-medium text-white-2">
                Kota Tanjungpinang
              </h1>
              <p className="text-lg text-white-3">
                Merupakan ibu kota Kepulauan Riau, yang menjadi fokus
                pengembangan model AI untuk prediksi kecepatan angin. Dengan
                tujuan memahami dan memprediksi pola angin, mendukung
                sektor-sektor rentan terhadap perubahan cuaca.
              </p>
            </div>
            <img src={img1} alt="" className="w-full md:w-[32rem] mx-auto" />
          </div>
          <img
            src={img2}
            alt=""
            className="w-full h-44 md:h-[16.75rem] md:w-[20rem] lg:h-auto object-cover md:object-none rounded-md md:hidden lg:block"
          />
        </div>

        <div className="flex flex-col lg:flex-row gap-y-8 py-12 xl:gap-x-32 items-start px-5">
          <img
            src={location}
            alt=""
            className="opacity-0 hidden lg:block"
            draggable="false"
          />
          <div className="flex flex-col gap-16">
            <div className="flex flex-col gap-6">
              <h1 className="text-3xl md:text-4xl font-medium text-white-2">
                Kabupaten Kep Anambas
              </h1>
              <p className="text-lg text-white-3">
                Sebagai wilayah dari Kepri, Kabupaten Kep Anambas menjadi
                sorotan dalam riset pengembangan model AI prediksi kecepatan
                angin. Tujuannya agar memahami dan memprediksi pola angin,
                mendukung sektor-sektor terdampak perubahan cuaca.
              </p>
            </div>
            <img src={img3} alt="" className="w-full md:w-[32rem] mx-auto" />
          </div>
          <img
            src={img4}
            alt=""
            className="w-full h-44 md:h-[16.75rem] md:w-[20rem] lg:h-auto object-cover md:object-none rounded-md md:hidden lg:block"
          />
        </div>
      </div>
    </section>
  );
}
