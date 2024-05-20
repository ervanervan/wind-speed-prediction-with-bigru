import React from "react";
import ReusableTable from "./ReusableTable";

export default function PredictNewData({ data, image }) {
  const headers = ["No", "Data Prediction"];
  const data_angin = data;

  return (
    <div className="container mx-auto">
      <div className="flex flex-col gap-10">
        <ReusableTable
          isNewData={true}
          isPredict={false}
          headers={headers}
          data={data_angin}
        />
        <div className="lg:w-full md:h-[36rem] lg:h-[44rem] overflow-hidden bg-transparent">
          <img
            className="w-full h-full object-fill object-center"
            src={image}
            alt=""
          />
        </div>
      </div>
    </div>
  );
}
