import React, { useState } from "react";

const Tabs = ({ tabs }) => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="container mx-auto px-5 py-20">
      <div className="flex gap-2 items-center justify-center">
        {tabs.map((tab, index) => (
          <button
            key={index}
            className={`px-7 py-2 font-medium text-xl focus:outline-none ${
              activeTab === index
                ? "bg-white-1 text-black-1 rounded-lg"
                : "text-white-1"
            }`}
            onClick={() => setActiveTab(index)}
          >
            {tab.icon && <tab.icon className="h-5 w-5 mr-2" />}
            {tab.label}
          </button>
        ))}
      </div>
      <div className="p-4">
        {tabs[activeTab] && <div>{tabs[activeTab].content}</div>}
      </div>
    </div>
  );
};

export default Tabs;
