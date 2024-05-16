import React from "react";
import ReusableTable from "./ReusableTable";

export default function AllData() {
  const headers = ["No", "Tanggal", "Data Actual"];
  const data = [
    ["1.", "01-01-2022", 3],
    ["2.", "02-01-2022", 5],
    ["3.", "03-01-2022", 4],
  ];

  return (
    <div className="container mx-auto">
      <ReusableTable headers={headers} data={data} />
    </div>
  );
}
