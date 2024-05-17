import React from "react";
import ReusableTable from "./ReusableTable";
import { useEffect } from "react";
import axios from "axios";

export default function PredictPastData({ data, model }) {
  const headers = ["No", "Tanggal", "Data Actual", "Data Prediction"];

  const data_angin = data;

  return (
    <div className="container mx-auto">
      <div className="px-3 text-white-2 font-semibold w-full text-center py-1">
        Mape: {model?.MAPE}% Akurasi: {model?.AKURASI}%
      </div>

      <ReusableTable
        type={"ff_x"}
        isPredict={true}
        headers={headers}
        data={data_angin}
      />
    </div>
  );
}
