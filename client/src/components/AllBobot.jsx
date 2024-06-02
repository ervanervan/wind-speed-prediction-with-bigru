import React from "react";
import ReusableTable from "./ReusableTable";

export default function AllBobot({ data, type }) {
  const headers = ["No", "Bobot"];

  const data_angin = data;

  return (
    <div className="container mx-auto">
      <ReusableTable type={type} headers={headers} data={data_angin} />
    </div>
  );
}
