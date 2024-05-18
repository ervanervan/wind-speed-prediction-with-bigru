import React from "react";

const ReusableTable = ({ headers, data, type, isPredict, isNewData }) => {
  return (
    <div className="overflow-x-auto lg:overflow-auto">
      {/* Loading State */}
      {!data ? (
        <div className="w-full md:max-w-2xl flex justify-center items-center h-64">
          <p className="text-lg font-medium text-white-2">Loading...</p>
        </div>
      ) : (
        <table className="w-full md:max-w-2xl">
          <thead>
            <tr>
              {headers.map((header, index) => (
                <th
                  key={index}
                  className="px-4 py-2 border-b border-dark-border text-left text-lg font-medium text-white-2"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <td
                  key={"num" + rowIndex}
                  className="px-4 py-2 border-b border-dark-border text-base text-white-2"
                >
                  {rowIndex + 1}
                </td>
                {!isNewData && (
                  <td
                    key={"tgl" + rowIndex}
                    className="px-4 py-2 border-b border-dark-border text-base text-white-2"
                  >
                    {row.Tanggal}
                  </td>
                )}
                {type === "ff_x" && (
                  <td
                    key={"ff_x" + rowIndex}
                    className="px-4 py-2 border-b border-dark-border text-base text-white-2"
                  >
                    {row.ff_x}
                  </td>
                )}
                {type === "ff_avg" && (
                  <td
                    key={"ff_avg" + rowIndex}
                    className="px-4 py-2 border-b border-dark-border text-base text-white-2"
                  >
                    {row.ff_avg}
                  </td>
                )}
                {isPredict && (
                  <td
                    key={"predicted" + rowIndex}
                    className="px-4 py-2 border-b border-dark-border text-base text-white-2"
                  >
                    {typeof row.predicted == "number"
                      ? row.predicted.toFixed(3)
                      : row.predicted}
                  </td>
                )}
                {isNewData && (
                  <td
                    key={"new data" + rowIndex}
                    className="px-4 py-2 border-b border-dark-border text-base text-white-2"
                  >
                    {row}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default ReusableTable;
