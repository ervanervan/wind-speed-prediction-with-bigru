import React from "react";
import ReusableTable from "./ReusableTable";
import { useEffect } from "react";
import axios from "axios";

export default function PredictPastData({ data, model, type }) {
  const headers = ["No", "Tanggal", "Data Actual", "Data Prediction"];

  const data_angin = data;

  return (
    <div className="container mx-auto">
      <div className="px-3 text-white-2 font-semibold w-full py-1">
        <div className="flex gap-6 bg-slate-800 px-4 py-2 rounded-md w-fit mx-auto mb-2">
          <span>Mape: {parseFloat(model?.MAPE).toFixed(2)}%</span>
          <span>Akurasi: {parseFloat(model?.AKURASI).toFixed(2)}%</span>
        </div>
      </div>

      <ReusableTable
        type={type}
        isPredict={true}
        headers={headers}
        data={data_angin}
      />
    </div>
  );
}
