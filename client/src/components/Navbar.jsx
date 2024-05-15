import React from "react";
import { Link } from "react-router-dom";
import logo from "../assets/logo.svg";

export default function Navbar() {
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
              className="px-8 py-3 text-white-1 font-medium text-base"
            >
              Anambas
            </Link>
            <Link
              to={"/tanjungpinang"}
              className="px-8 py-3 text-white-1 font-medium text-base"
            >
              Tanjungpinang
            </Link>
          </div>
        </div>
        <div className="text-white-1 md:hidden">Menu</div>
      </nav>
    </header>
  );
}
