import React from "react";

const FF_X_ANB = () => {
  //   const [tab, setTab] = useState("all_data");
  return (
    <>
      <div className="container mx-auto">
        <h1 className="text-2xl font-bold my-4">
          Predict Wind Speed FF X ANAMBAS
        </h1>
        <div className="flex gap-5 items-center">
          <button
            className="bg-gray-200 p-2 rounded-md"
            onClick={() => {
              setTab("all_data");
            }}
          >
            ALL DATA
          </button>
          <button
            className="bg-gray-200 p-2 rounded-md"
            onClick={() => {
              setTab("past_data");
            }}
          >
            PREDICT PAST DATA
          </button>
          <button
            className="bg-gray-200 p-2 rounded-md"
            onClick={() => {
              setTab("new_data");
            }}
          >
            PREDICT NEW DATA
          </button>
        </div>
      </div>
    </>
  );
};

export default FF_X_ANB;
