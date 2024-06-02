import React from "react";
import ReusableTable from "./ReusableTable";

export default function PredictPastData({ data, model, type, images }) {
  const headers = ["No", "Tanggal", "Data Actual", "Data Prediction"];

  const data_angin = data;

  return (
    <div className="container mx-auto">
      <div className="px-3 text-white-2 font-semibold w-full py-1">
        <div className="flex gap-6 bg-slate-800 px-4 py-2 rounded-md w-fit mx-auto mb-2">
          <span>
            Mape Training:{" "}
            {parseFloat(
              model ? model["Training Performance"]?.MAPE : 0
            ).toFixed(2)}
            %
          </span>

          <span>
            Akurasi Training:{" "}
            {parseFloat(
              model ? model["Training Performance"]?.Accuracy : 0
            ).toFixed(2)}
            %
          </span>

          <span>
            Mape Testing:{" "}
            {parseFloat(model ? model["Testing Performance"]?.MAPE : 0).toFixed(
              2
            )}
            %
          </span>
          <span>
            Akurasi Testing:{" "}
            {parseFloat(
              model ? model["Testing Performance"]?.Accuracy : 0
            ).toFixed(2)}
            %
          </span>
        </div>
        {images?.map((item, idx) => (
          <img key={idx} src={item} alt="" className="my-7 rounded-md" />
        ))}
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
