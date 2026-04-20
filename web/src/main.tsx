import React from "react";
import ReactDOM from "react-dom/client";

// Inter variable font -- self-hosted via @fontsource-variable/inter.
// Loading here (not index.html) means Vite bundles the .woff2 and the
// font matches the rest of the asset pipeline (cache-busted filenames,
// production compression).
import "@fontsource-variable/inter";

// Global chrome. tokens.css is re-imported by app.css, so one import
// wires the whole system.
import "./styles/app.css";

import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
