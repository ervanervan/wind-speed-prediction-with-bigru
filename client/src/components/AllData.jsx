import React from "react";
import ReusableTable from "./ReusableTable";

export default function AllData({ data }) {
  const headers = ["No", "Tanggal", "Data Actual"];
  const data_angin = data;

  return (
    <div className="container mx-auto">
      <ReusableTable
        type={"ff_x"}
        isPredict={false}
        headers={headers}
        data={data_angin}
      />
    </div>
  );
}
