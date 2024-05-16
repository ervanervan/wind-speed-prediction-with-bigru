import React from "react";
import IconData from "../assets/icons/IconData";
import TabsData from "../components/TabsData";
import AllData from "../components/AllData";
import PredictPastData from "../components/PredictPastData";
import PredictNewData from "../components/PredictNewData";

const FF_X_TPI = () => {
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
          Kecepatan Angin Maksimum
        </h1>
        <TabsData tabs={tabs} />
      </div>
    </>
  );
};

export default FF_X_TPI;
