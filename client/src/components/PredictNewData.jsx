import React, { useState } from "react";
import ReusableTable from "./ReusableTable";
import axios from "axios";

export default function PredictNewData({ type, image }) {
  const headers = ["No", "Data Prediction"];
  // const data_angin = data;

  const [inputData, setInputData] = useState("");
  const [data, setData] = useState([]);
  const handleInputChange = (e) => {
    setInputData(e.target.value);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();

    console.log("Input Data:", inputData);
    let res;
    if (type === "ff_avg_anb") {
      res = await axios.post("http://localhost:5000/ff-avg-anb-input-90", {
        input: inputData,
      });
    } else if (type === "ff_x_anb") {
      res = await axios.post("http://localhost:5000/ff-x-anb-input-90", {
        input: inputData,
      });
    } else if (type === "ff_avg_tpi") {
      res = await axios.post("http://localhost:5000/ff-avg-tpi-input-90", {
        input: inputData,
      });
    } else if (type === "ff_x_tpi") {
      res = await axios.post("http://localhost:5000/ff-avg-tpi-input-90", {
        input: inputData,
      });
    }
    setData(res.data.predicted);
  };

  return (
    <div className="container mx-auto">
      <div className="flex flex-col gap-10">
        Form Input
        <form onSubmit={handleFormSubmit} className="md:max-w-2xl w-full">
          <label
            htmlFor="inputData"
            className="block text-base md:text-lg font-medium text-gray-400"
          >
            Masukkan Data Kecepatan Angin 5 Hari Sebelumnya
          </label>
          <div className="flex flex-col md:flex-row items-start justify-between gap-2 mt-2">
            <input
              type="text"
              name="inputData"
              id="inputData"
              value={inputData}
              onChange={handleInputChange}
              className="block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-base"
              placeholder="Contoh: 3,1,2,3,4"
            />
            <button
              type="submit"
              className="inline-flex px-7 py-2 font-medium shrink-0 text-base border rounded-lg border-dark-1 hover:bg-gray-800 bg-transparent transition-all duration-300 text-white-1"
            >
              Prediksi
            </button>
          </div>
        </form>
        {data.length > 0 && (
          <ReusableTable
            isNewData={true}
            isPredict={false}
            headers={headers}
            data={data}
          />
        )}
        {/* <div className="lg:w-full md:h-[36rem] lg:h-[44rem] overflow-hidden bg-transparent">
          <img
            className="w-full h-full object-fill object-center rounded-md"
            src={image}
            alt=""
          />
        </div> */}
      </div>
    </div>
  );
}
