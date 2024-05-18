import React, { useState } from "react";
import { Link } from "react-router-dom";
import logo from "../assets/logo.svg";
import logoWhite from "../assets/logo-white.svg";
import IconMenu from "../assets/icons/IconMenu";
import IconClose from "../assets/icons/IconClose";
import gradientLCL from "../assets/gradient-l-c-l.svg";
import gradientFBR from "../assets/gradient-f-b-r.svg";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    setIsOpen((prevState) => !prevState);
  };

  return (
    <header className="container mx-auto py-6 px-5">
      <nav className="flex items-center justify-between py-2">
        <Link to={"/"}>
          <h1 className="text-white-1 font-medium text-base">
            Ervan Kurniawan
          </h1>
        </Link>
        <div className="md:flex items-center md:gap-x-12 lg:gap-x-20 xl:gap-x-60 hidden">
          <Link to={"/"}>
            <img src={logo} alt="logo" />
          </Link>
          <div className="flex lg:gap-8">
            <Link
              to={"/anambas"}
              className="px-8 py-3 border border-transparent rounded-md hover:border-dark-1 transition-all duration-300 text-white-1 font-medium text-base"
            >
              Anambas
            </Link>
            <Link
              to={"/tanjungpinang"}
              className="px-8 py-3 border border-transparent rounded-md hover:border-dark-1 transition-all duration-300 text-white-1 font-medium text-base"
            >
              Tanjungpinang
            </Link>
          </div>
        </div>
        <button className="md:hidden" onClick={handleToggle}>
          <IconMenu className="size-7 stroke-white-2" />
        </button>

        <div
          className={`fixed h-screen w-full top-0 right-0 z-50 bg-dark text-textBlack flex flex-col items-start justify-start shadow-lg gap-y-4 px-5 pt-[7.5rem] pb-8 transition-all duration-500 transform ${
            isOpen
              ? "translate-x-[0%] ease-in-out"
              : "-translate-x-[100%] ease-in-out"
          }`}
        >
          <div className="text-textBlack flex flex-col items-start justify-center w-full gap-y-5 sm:px-14">
            <Link
              to={"/anambas"}
              className="py-3 text-white-1 font-medium text-xl"
            >
              Anambas
            </Link>
            <hr className="border[.5px] border-bgBlack2 w-full" />

            <Link
              to={"/tanjungpinang"}
              className="py-3 text-white-1 font-medium text-xl"
            >
              Tanjungpinang
            </Link>
            <hr className="border[.5px] border-bgBlack2 w-full" />
          </div>

          <img
            src={gradientLCL}
            alt=""
            className="absolute -top-96 left-0 -z-10"
          />
          <img
            src={gradientFBR}
            alt=""
            className="absolute bottom-0 right-0 -z-10"
          />

          <Link to={"/"} className="absolute top-8 left-5 sm:left-16">
            <img src={logoWhite} alt="" className="w-32" />
          </Link>
          <button
            className="absolute top-7 right-4 sm:right-16"
            onClick={handleToggle}
          >
            <IconClose className="size-9 stroke-white-2" />
          </button>
        </div>
      </nav>
    </header>
  );
}
