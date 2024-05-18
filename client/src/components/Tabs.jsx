import React, { useState } from "react";

const Tabs = ({ tabs }) => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="container mx-auto">
      <div className="flex gap-2 items-center justify-center w-full px-5 py-6 overflow-x-scroll scrollbar-hide">
        {tabs.map((tab, index) => (
          <button
            key={index}
            className={`px-7 py-2 font-medium shrink-0 text-xl border border-transparent rounded-lg hover:border-dark-1 transition-all duration-300 ${
              activeTab === index
                ? "bg-white-1 text-black-1 rounded-lg"
                : "text-white-1"
            }`}
            onClick={() => setActiveTab(index)}
          >
            <div className="flex items-center">
              {tab.icon && <tab.icon className="h-6 w-6 mr-2" />}
              {tab.label}
            </div>
          </button>
        ))}
      </div>
      <div>{tabs[activeTab] && <div>{tabs[activeTab].content}</div>}</div>
    </div>
  );
};

export default Tabs;
