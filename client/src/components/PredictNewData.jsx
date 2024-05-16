import React from "react";
import ReusableTable from "./ReusableTable";

export default function PredictNewData() {
  const headers = ["No", "Data Prediction"];
  const data = [
    ["1.", 3.25],
    ["2.", 5.15],
    ["3.", 4.35],
  ];
  return (
    <div className="container mx-auto">
      <ReusableTable headers={headers} data={data} />
    </div>
  );
}
