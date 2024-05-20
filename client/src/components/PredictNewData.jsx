import React from "react";
import ReusableTable from "./ReusableTable";

export default function PredictNewData({ data, image }) {
  const headers = ["No", "Data Prediction"];
  const data_angin = data;

  return (
    <div className="container mx-auto">
      <div className="flex justify-between">
        <ReusableTable
          isNewData={true}
          isPredict={false}
          headers={headers}
          data={data_angin}
        />
        {/* <div>
          <img
            className="w-[55%]"
            src="/forecasting_Bidirectional_GRU_FF_X_ANAMBAS.jpeg"
            alt=""
          />
        </div> */}
      </div>
    </div>
  );
}
