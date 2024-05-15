import "./App.css";
import { Route, Routes } from "react-router-dom";
import Homepage from "./pages/Homepage";
import FF_X_ANB from "./pages/FF_X_ANB";
import TanjungPinangPage from "./pages/TanjungPinangPage";
import AnambasPage from "./pages/AnambasPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Homepage />}></Route>
      <Route path="/tanjungpinang" element={<TanjungPinangPage />}></Route>
      <Route path="/anambas" element={<AnambasPage />}></Route>
      <Route path="/ff-x-anb" element={<FF_X_ANB />}></Route>
    </Routes>
  );
}

export default App;
