import React, { useEffect, useState } from "react";

import axios from "axios";

import AllData from "../components/AllData";
import AllBias from "../components/AllBias";
import TabsData from "../components/TabsData";
import AllBobot from "../components/AllBobot";
import PredictNewData from "../components/PredictNewData";
import PredictPastData from "../components/PredictPastData";

import IconData from "../assets/icons/IconData";
import LossAVGANB from "../assets/images/Loss_Plot_Bidirectional_GRU_FF_AVG_TANJUNGPINANG.jpeg";
import ActPreAVGANB from "../assets/images/Actual_and_Prediction_Bidirectional_GRU_FF_AVG_TANJUNGPINANG.jpeg";

const FF_AVG_TPI = () => {
  const [data, setData] = useState([]);
  const [newData, setNewData] = useState([]);
  const [modelPerformance, setModelPerformance] = useState({});
  const [bobot, setBobot] = useState([]);
  const [bias, setBias] = useState([]);

  const tabs = [
    {
      label: "All Data",
      icon: IconData,
      content: <AllData type={"ff_avg"} data={data ? data : []} />,
    },
    {
      label: "Predict Past Data",
      icon: IconData,
      content: (
        <PredictPastData
          images={[LossAVGANB, ActPreAVGANB]}
          type={"ff_avg"}
          model={modelPerformance}
          data={data ? data : []}
        />
      ),
    },
    {
      label: "Predict New Data",
      icon: IconData,
      content: <PredictNewData type={"ff_avg_tpi"} />,
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
      const response = await axios.get("http://localhost:5000/ff-avg-tpi");
      const responseOriginal = await axios.get(
        "http://localhost:5000/ff-avg-tpi-original"
      );

      const responseModelPerformance = await axios.get(
        "http://localhost:5000/ff-avg-tpi-performance"
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
    //     "http://localhost:5000/ff-avg-tpi-forcasting"
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
          Kecepatan Angin Rata-rata (m/s)
        </h1>
        <TabsData tabs={tabs} />
      </div>
    </>
  );
};

export default FF_AVG_TPI;
