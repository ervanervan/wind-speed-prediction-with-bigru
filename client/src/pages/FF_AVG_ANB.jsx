import React from "react";
import IconData from "../assets/icons/IconData";
import TabsData from "../components/TabsData";
import AllData from "../components/AllData";
import PredictPastData from "../components/PredictPastData";
import PredictNewData from "../components/PredictNewData";

const FF_AVG_ANB = () => {
  const tabs = [
    {
      label: "All Data",
      icon: IconData,
      content: <AllData />,
    },
    {
      label: "Predict Past Data",
      icon: IconData,
      content: <PredictPastData />,
    },
    {
      label: "Predict New Data",
      icon: IconData,
      content: <PredictNewData />,
    },
  ];

  return (
    <>
      <div className="px-5 py-4">
        <h1 className="text-3xl text-white-1 font-semibold">
          Kecepatan Angin Rata-rata
        </h1>
        <TabsData tabs={tabs} />
      </div>
    </>
  );
};

export default FF_AVG_ANB;
