import React, { useState, useEffect } from "react";
import WorkspaceContainer from "./components/WorkspaceContainer";
import { ComponentType } from "./models/Component";
import Spinner from "./components/common/Spinner";
import cx from "./utils/cx";
import { BASE_URL } from "./services/mbx";
import "./App.css";

export function App() {
  const [initData, setInitData] = useState(null);
  const [componentMode, setComponentMode] = useState(ComponentType.PCA);

  useEffect(() => {
    (async () => {
      // sending a timestamp at startup to avoid cache,
      // this first request should be cheap since we only get the workspace id
      // and a few other initial parameters (such as initial sample window size),
      // in the next requests we send the workspace id in the request args
      // so if the data for that workspace id is already cached we can reuse it
      const initTimestamp = new Date().toISOString();
      const r = await fetch(`${BASE_URL}/init?t=${initTimestamp}`);
      const data = await r.json();
      setInitData(data);
    })();
  }, []);

  const onBtnComponentModeClick = () => {
    setComponentMode(componentMode.toggle());
  };

  const otherComponentMode = componentMode.toggle();
  const componentModeText = `${componentMode.modeName} mode`;
  const toggleText = `Switch to ${otherComponentMode.modeName} mode`;

  const body =
    initData == null ? (
      <Spinner />
    ) : (
      <WorkspaceContainer
        id={initData.workspace_id}
        sampleWindowSize={initData.sample_window_size}
        maxSampleWindowSize={initData.max_sample_window_size}
        minSampleWindowSize={initData.min_sample_window_size}
        componentMode={componentMode}
      />
    );

  return (
    <div className="mbx-app">
      <header className="mbx-app-header">
        <div className="mbx-app-header-title">Embedding Explorer</div>
        <div>{componentModeText}</div>
        <button
          className={cx(["btn-component-mode", "btn-secondary", "btn"])}
          onClick={onBtnComponentModeClick}
        >
          {toggleText}
        </button>
      </header>
      {body}
    </div>
  );
}
