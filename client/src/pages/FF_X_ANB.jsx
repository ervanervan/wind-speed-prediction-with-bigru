import React, { useEffect, useState } from "react";

import axios from "axios";

import AllData from "../components/AllData";
import AllBias from "../components/AllBias";
import TabsData from "../components/TabsData";
import AllBobot from "../components/AllBobot";
import PredictNewData from "../components/PredictNewData";
import PredictPastData from "../components/PredictPastData";

import IconData from "../assets/icons/IconData";
import LossAVGANB from "../assets/images/Loss_Plot_Bidirectional_GRU_FF_X_ANAMBAS.jpeg";
import ActPreAVGANB from "../assets/images/Actual_and_Prediction_Bidirectional_GRU_FF_X_ANAMBAS.jpeg";

const FF_X_ANB = () => {
  const [data, setData] = useState([]);
  const [newData, setNewData] = useState([]);
  const [modelPerformance, setModelPerformance] = useState({});
  const [bobot, setBobot] = useState([]);
  const [bias, setBias] = useState([]);

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
          images={[LossAVGANB, ActPreAVGANB]}
          type={"ff_x"}
          model={modelPerformance}
          data={data ? data : []}
        />
      ),
    },
    {
      label: "Predict New Data",
      icon: IconData,
      content: <PredictNewData type={"ff_x_anb"} />,
    },
    {
      label: "All Bobot",
      icon: IconData,
      content: <AllBobot type="bobot" data={bobot ? bobot : []} />,
    },
    {
      label: "All Bias",
      icon: IconData,
      content: <AllBias type="bias" data={bias ? bias : []} />,
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
      const bobotData = responseModelPerformance.data.Bobot;
      const biasData = responseModelPerformance.data.Bias;

      original.map((item, idx) => {
        if (idx < 5) {
          item.predicted = "";
        } else {
          item.predicted = predicted[idx - 5];
        }
      });

      setData(original);
      setModelPerformance(responseModelPerformance.data);
      setBobot(bobotData);
      setBias(biasData);
    }

    // async function getNewData() {
    //   const response = await axios.get(
    //     "http://localhost:5000/ff-x-anb-forcasting"
    //   );
    //   setNewData(response.data.predicted);
    // }
    getData();
    // getNewData();
  }, []);

  return (
    <>
      <div className="px-5 py-4">
        <h1 className="text-3xl text-white-1 font-semibold">
          Kecepatan Angin Maksimum (m/s)
        </h1>
        <TabsData tabs={tabs} />
      </div>
    </>
  );
};

export default FF_X_ANB;
