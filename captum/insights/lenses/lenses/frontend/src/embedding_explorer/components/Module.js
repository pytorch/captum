import React from "react";
import { SamplesWindow } from "./Samples";
import { ComponentType } from "../models/Component";
import ComponentBars from "./ComponentBars";
import "./Module.css";

function ComponentBar(props) {
  return (
    <div
      className="module-component-bar"
      style={{
        width: `${props.width}px`,
        minHeight: `${props.height}px`,
      }}
    >
      <div
        className="module-component-bar-bg"
        style={{
          width: `${props.width}px`,
          minHeight: `${props.height}px`,
        }}
      ></div>
      <div
        className="module-component-bar-fg"
        style={{
          width: `${props.value * props.width}px`,
          minHeight: `${props.height}px`,
        }}
      ></div>
    </div>
  );
}

function Component(props) {
  const component = props.component;
  const windowSize = 3;
  const id = component.id;
  const ratioPercent = Math.round(component.value * 100);
  let description = "";
  if (component.type === ComponentType.PCA) {
    const txtRatio = ratioPercent < 1 ? "< 1%" : `${ratioPercent}%`;
    description = `PC ${id + 1} explains ${txtRatio} of the variance`;
  } else if (component.type === ComponentType.ICA) {
    description = `IC ${id + 1}`;
  }

  const onClick = (e) => props.onClick && props.onClick(e);
  const onViewDetailClick = () => props.onViewDetail && props.onViewDetail();

  const [barWidth, barHeight] = [200, 12];
  let [samplesLo, samplesHi] = [null, null];

  if (component.sampleBins.length > 0) {
    const firstSampleBin = component.sampleBins[0];
    samplesLo = (
      <div className="module-component-samples">
        <SamplesWindow
          sampleBin={firstSampleBin}
          size={windowSize}
          anchor="begin"
        />
      </div>
    );

    const lastSampleBin = component.sampleBins[component.sampleBins.length - 1];
    samplesHi = (
      <div className="module-component-samples">
        <SamplesWindow
          sampleBin={lastSampleBin}
          size={windowSize}
          anchor="end"
        />
      </div>
    );
  }

  return (
    <div className="module-component-overview">
      <div className="module-component-overview-top">
        <div>{description}</div>
        <div
          className="module-component-view-detail"
          onClick={onViewDetailClick}
        >
          View Detail
        </div>
      </div>
      <div className="module-component-overview-bottom" onClick={onClick}>
        {samplesLo}
        <ComponentBar
          value={component.value}
          width={barWidth}
          height={barHeight}
        />
        {samplesHi}
      </div>
    </div>
  );
}

function Module(props) {
  const module = props.module;
  const components = module.getComponents(props.componentMode);
  const componentsView = components.map((component, i) => {
    const onViewDetail = () =>
      props.onViewComponentDetail && props.onViewComponentDetail(component);
    const onClick = (e) =>
      props.onComponentClick && props.onComponentClick(component, e);
    return (
      <Component
        key={i}
        component={component}
        onViewDetail={onViewDetail}
        onClick={onClick}
        explorerSelectionContext={props.explorerSelectionContext}
        workspaceSelectionContext={props.workspaceSelectionContext}
      />
    );
  });

  return (
    <div className="module">
      <div className="module-header">
        <div className="module-header-title">{module.name}</div>
        <ComponentBars
          direction="row"
          components={components}
          explorerSelectionContext={props.explorerSelectionContext}
          workspaceSelectionContext={props.workspaceSelectionContext}
          componentMode={props.componentMode}
          updateColor={true}
        />
      </div>
      {componentsView}
    </div>
  );
}

export default Module;
