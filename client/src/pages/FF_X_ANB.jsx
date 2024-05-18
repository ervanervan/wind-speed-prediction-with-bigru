import React, { useEffect, useState } from "react";
import IconData from "../assets/icons/IconData";
import TabsData from "../components/TabsData";
import AllData from "../components/AllData";
import PredictPastData from "../components/PredictPastData";
import PredictNewData from "../components/PredictNewData";
import axios from "axios";

const FF_X_ANB = () => {
  const [data, setData] = useState([]);
  const [newData, setNewData] = useState([]);
  const [modelPerformance, setModelPerformance] = useState({});
  const tabs = [
    {
      label: "All Data",
      icon: IconData,
      content: <AllData type={"ff_x"} data={data ? data : []} />,
    },
    {
      label: "Predict Past Data",
      icon: IconData,
      content: (
        <PredictPastData
          type={"ff_x"}
          model={modelPerformance}
          data={data ? data : []}
        />
      ),
    },
    {
      label: "Predict New Data",
      icon: IconData,
      content: <PredictNewData data={newData ? newData : []} />,
    },
  ];

  useEffect(() => {
    async function getData() {
      const response = await axios.get("http://localhost:5000/ff-x-anb");
      const responseOriginal = await axios.get(
        "http://localhost:5000/ff-x-anb-original"
      );

      const responseModelPerformance = await axios.get(
        "http://localhost:5000/ff-x-anb-performance"
      );

      // console.log(response.data);
      const original = responseOriginal.data;
      const predicted = response.data.predicted;

      original.map((item, idx) => {
        if (idx < 5) {
          item.predicted = "";
        } else {
          item.predicted = predicted[idx - 5];
        }
      });

      setData(original);
      setModelPerformance(responseModelPerformance.data);
    }

    async function getNewData() {
      const response = await axios.get(
        "http://localhost:5000/ff-x-anb-forcasting"
      );
      setNewData(response.data.predicted);
    }
    getData();
    getNewData();
  }, []);
  return (
    <>
      <div className="px-5 py-4">
        <h1 className="text-3xl text-white-1 font-semibold">
          Kecepatan Angin Maksimum
        </h1>
        <TabsData tabs={tabs} />
      </div>
    </>
  );
};

export default FF_X_ANB;
