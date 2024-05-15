import React from "react";
import logoWhite from "../assets/logo-white.svg";
import gradientFL from "../assets/gradient-f-b-l.svg";
import gradientFR from "../assets/gradient-f-b-r.svg";

export default function Footer() {
  return (
    <footer className="relative">
      <img
        src={gradientFL}
        alt=""
        className="absolute bottom-0 left-0 -z-10 hidden md:block"
      />
      <img
        src={gradientFR}
        alt=""
        className="absolute bottom-0 right-0 -z-10 hidden md:block"
      />
      <div className="container mx-auto pt-10">
        <div className="flex flex-col gap-y-6 md:flex-row md:items-end justify-between p-5 md:py-10">
          <div className="flex flex-col gap-3">
            <img src={logoWhite} alt="" className="w-28 md:w-36" />
            <p className="text-white-3 text-sm">
              &copy; 2024 Made by Ervan Kurniawan
            </p>
          </div>
          <div className="flex flex-col gap-y-1 md:flex-row gap-x-24 text-white-3 text-sm">
            <p>Teknik Informatika 2020</p>
            <p>Fakultas Teknik</p>
            <p>Universitas Maritim Raja Ali haji</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
