import React from "react";
import IconData from "../assets/icons/IconData";
import TabsData from "../components/TabsData";

const FF_AVG_ANB = () => {
  const tabs = [
    {
      label: "All Data",
      icon: IconData,
      content: <div>Content for Tab 1</div>,
    },
    {
      label: "Predict Past Data",
      icon: IconData,
      content: <div>Content for Tab 2</div>,
    },
    {
      label: "Predict New Data",
      icon: IconData,
      content: <div>Content for Tab 3</div>,
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
