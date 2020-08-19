import React from "react";
import { Lenses } from "./common/models/lens";
import { App as EmbeddingExplorer } from "./embedding_explorer/App";
import { getActiveLens } from "./common/services/lens";
import { CaptumLogo } from "./common/icons";

import "./App.css";

export function App() {
  const [lens, setLens] = React.useState(Lenses.PLACEHOLDER);

  React.useEffect(() => {
    getActiveLens().then((lens) => {
      setLens(lens.name);
    });
  }, []);

  const LensRoot = () => {
    switch (lens) {
      case Lenses.EMBEDDING_EXPLORER:
        return <EmbeddingExplorer />;
    }

    return <></>;
  };

  return (
    <div id="app-container">
      <div id="app-header">
        <CaptumLogo id="app-header--logo" />
        <h1>Captum Insights</h1>
      </div>
      <div id="app-content">
        <LensRoot />
      </div>
    </div>
  );
}
