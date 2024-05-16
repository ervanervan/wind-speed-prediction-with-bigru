import React from "react";
import ReusableTable from "./ReusableTable";

export default function PredictPastData() {
  const headers = ["No", "Tanggal", "Data Actual", "Data Prediction"];
  const data = [
    ["1.", "01-01-2022", 3, 2.75],
    ["2.", "02-01-2022", 5, 5.25],
    ["3.", "03-01-2022", 4, 4.25],
  ];
  return (
    <div className="container mx-auto">
      <ReusableTable headers={headers} data={data} />
    </div>
  );
}
