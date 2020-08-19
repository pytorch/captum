import React from "react";
import Samples from "./Samples";
import { ComponentType } from "../models/Component";
import ComponentBars from "./ComponentBars";
import "./ComponentDetail.css";

function ComponentDetail(props) {
  const component = props.component;
  const id = component.id;
  const ratioPercent = Math.round(component.value * 100);
  let description = "";
  if (component.type === ComponentType.PCA) {
    const txtRatio = ratioPercent < 1 ? "< 1%" : `${ratioPercent}%`;
    description = `PC ${id + 1} explains ${txtRatio} of the variance`;
  }

  return (
    <div className="component-detail">
      <div className="component-detail-header">
        <div className="component-detail-module-name">
          {component.module.name}
        </div>
        <ComponentBars
          components={component.module.getComponents(component.type)}
          direction="row"
          explorerSelectionContext={props.explorerSelectionContext}
        />
      </div>
      <div className="component-detail-description">{description}</div>
      <Samples component={component} windowSize={props.sampleWindowSize} />
    </div>
  );
}

export default ComponentDetail;
