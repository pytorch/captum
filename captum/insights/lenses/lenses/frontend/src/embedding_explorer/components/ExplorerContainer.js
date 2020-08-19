import React, { useState, useEffect } from "react";
import { ReactComponent as LeftArrow } from "../icons/leftArrow.svg";
import Explorer from "../models/Explorer";
import Spinner from "./common/Spinner";
import Modules from "./Modules";
import Card from "./common/Card";
import ComponentDetail from "./ComponentDetail";
import Module from "./Module";
import CorrelatedComponents from "./CorrelatedComponents";
import cx from "../utils/cx";
import { BASE_URL } from "../services/mbx";
import "./ExplorerContainer.css";

function TabHeader(props) {
  const correlationMode = props.correlationMode;
  return (
    <div className="explorer-tab-view-header">
      <div
        className={cx([
          "explorer-tab-view-header-tab",
          correlationMode ? null : "explorer-tab-view-header-tab-active",
        ])}
        onClick={() =>
          props.onActiveTabUpdate && props.onActiveTabUpdate(false)
        }
      >
        Samples
      </div>
      <div
        className={cx([
          "explorer-tab-view-header-tab",
          correlationMode ? "explorer-tab-view-header-tab-active" : null,
        ])}
        onClick={() => props.onActiveTabUpdate && props.onActiveTabUpdate(true)}
      >
        Correlations
      </div>
    </div>
  );
}

function ExplorerContainer(props) {
  const [explorer, setExplorer] = useState(null);
  const [selectedModuleIds, setSelectedModuleIds] = useState([]);
  const [selectionContext, setSelectionContext] = useState({});
  const [correlationMode, setCorrelationMode] = useState(false);

  const maxSampleWindowSize = props.maxSampleWindowSize ?? 6;
  const minSampleWindowSize = props.minSampleWindowSize ?? 3;
  const initialSampleWindowSize =
    props.sampleWindowSize == null
      ? 3
      : Math.min(
          Math.max(props.sampleWindowSize, minSampleWindowSize),
          maxSampleWindowSize,
        );
  const [sampleWindowSize, setSampleWindowSize] = useState(
    initialSampleWindowSize,
  );

  useEffect(() => {
    setExplorer(null);
    (async () => {
      const workspace = props.workspace;
      const params = new URLSearchParams({
        workspace_id: workspace.id,
        id: props.id,
      });
      const url = `${BASE_URL}/explorer?${params.toString()}`;
      const r = await fetch(url);
      const data = await r.json();
      const explorer = Explorer.fromData(data, workspace);
      setExplorer(explorer);
    })();
  }, [props.workspace, props.id]);

  const onViewComponentDetail = (component) => {
    setSelectionContext({
      selectedComponent: component,
    });
  };

  useEffect(() => {
    setSelectionContext({});
  }, [props.componentMode]);

  const onCorrelatedComponentClick = (component) => {
    if (component.equals(selectionContext.selectedComponent)) {
      setSelectionContext({
        selectedComponent: null,
      });
    } else {
      setSelectionContext({
        selectedComponent: component,
      });
    }
  };

  const onComponentClick = (component, e) =>
    props.onComponentClick && props.onComponentClick(component, e);

  const onModuleListModuleClick = (module) => {
    const i = selectedModuleIds.indexOf(module.id);
    if (i === -1) {
      setSelectedModuleIds(selectedModuleIds.concat([module.id]));
    } else {
      setSelectedModuleIds(
        selectedModuleIds.slice(0, i).concat(selectedModuleIds.slice(i + 1)),
      );

      // deselect component if its module is removed from the list of selected modules
      const selectedComponent = selectionContext.selectedComponent;
      if (selectedComponent?.module?.id === module.id) {
        setSelectionContext({ selectedComponent: null });
      }
    }
  };

  let body;
  if (explorer == null) {
    body = <Spinner />;
  } else {
    const selectedModules = [];
    explorer.modules.forEach((module) => {
      if (selectedModuleIds.indexOf(module.id) !== -1) {
        selectedModules.push(module);
      }
    });
    const left = (
      <div className="explorer-body-left">
        <Card title={`Layers (${explorer.modules.length})`}>
          <Modules
            modules={explorer.modules}
            selectedModuleIds={selectedModuleIds}
            direction="column"
            componentDirection="row"
            onModuleClick={onModuleListModuleClick}
            componentMode={props.componentMode}
            updateColor={false}
            explorerSelectionContext={selectionContext}
            workspaceSelectionContext={props.workspaceSelectionContext}
          />
        </Card>
      </div>
    );

    let rightContent;
    let rightTitleText;
    let tabContent;

    let selectedComponent = selectionContext.selectedComponent;

    // TODO generalize if we need multiple tabs
    const onActiveTabUpdate = (correlationMode) =>
      setCorrelationMode(correlationMode);
    const tabHeader = (
      <TabHeader
        onActiveTabUpdate={onActiveTabUpdate}
        correlationMode={correlationMode}
      />
    );

    if (selectedComponent == null) {
      rightTitleText = `Selected Layers (${selectedModules.length})`;
    } else {
      const module = selectedComponent.module;
      selectedComponent = module.getComponents(props.componentMode)[
        selectedComponent.id
      ];
      rightTitleText = [
        module.name,
        `${selectedComponent.type.componentPrefix} ${selectedComponent.id + 1}`,
      ].join(" / ");
    }

    if (selectedComponent != null && !correlationMode) {
      tabContent = (
        <ComponentDetail
          component={selectedComponent}
          sampleWindowSize={sampleWindowSize}
          explorerSelectionContext={selectionContext}
        />
      );
    } else {
      if (correlationMode) {
        tabContent = (
          <CorrelatedComponents
            modules={selectedModules}
            componentMode={props.componentMode}
            onComponentClick={onCorrelatedComponentClick}
            explorerSelectionContext={selectionContext}
          />
        );
      } else {
        tabContent = selectedModules.map((module, i) => (
          <Module
            key={i}
            module={module}
            componentMode={props.componentMode}
            onViewComponentDetail={onViewComponentDetail}
            onComponentClick={onComponentClick}
            explorerSelectionContext={selectionContext}
            workspaceSelectionContext={props.workspaceSelectionContext}
          />
        ));
      }
    }

    rightContent = (
      <div className="explorer-tab-view">
        {tabHeader}
        <div className="explorer-tab-view-body">{tabContent}</div>
      </div>
    );

    const onBackClick = () => setSelectionContext({});

    const selectionIsEmpty =
      selectionContext.selectedComponent == null &&
      selectionContext.selectedModule == null;
    const btnBack = (
      <button
        className={cx(["btn-icon", "btn-back"])}
        onClick={onBackClick}
        disabled={selectionIsEmpty}
      >
        <LeftArrow />
      </button>
    );

    const onMoreSamplesClick = () =>
      setSampleWindowSize(Math.min(sampleWindowSize + 1, maxSampleWindowSize));

    const onFewerSamplesClick = () =>
      setSampleWindowSize(Math.max(sampleWindowSize - 1, minSampleWindowSize));

    const btnMoreSamples = (
      <button
        className={cx(["btn-icon", "btn-more-samples"])}
        disabled={sampleWindowSize === maxSampleWindowSize}
        onClick={onMoreSamplesClick}
      >
        +
      </button>
    );

    const btnFewerSamples = (
      <button
        className={cx(["btn-icon", "btn-fewer-samples"])}
        disabled={sampleWindowSize === minSampleWindowSize}
        onClick={onFewerSamplesClick}
      >
        -
      </button>
    );

    const rightTitle = (
      <div className="explorer-detail-title">
        {btnBack}
        <div>{rightTitleText}</div>
        <div
          className="explorer-detail-title-btn-group"
          style={{
            visibility:
              selectionContext.selectedComponent != null && !correlationMode
                ? "visible"
                : "hidden",
          }}
        >
          {btnFewerSamples}
          {btnMoreSamples}
        </div>
      </div>
    );
    const right = (
      <div className="explorer-body-right">
        <Card title={rightTitle}>{rightContent}</Card>
      </div>
    );

    body = (
      <div className="explorer-body">
        {left}
        {right}
      </div>
    );
  }

  const onCompareClick = () => props.onCompare && props.onCompare(props.idx);
  const onCloseClick = () => props.onClose && props.onClose(props.idx);

  const title = explorer == null ? `Explorer ID ${props.id}` : explorer.name;
  const btnCompare = (
    <button
      className={cx(["btn", "btn-secondary", "btn-compare"])}
      onClick={onCompareClick}
    >
      Compare
    </button>
  );
  const btnClose = (
    <button className={cx(["btn-icon", "btn-close"])} onClick={onCloseClick}>
      {"\u2715"}
    </button>
  );

  // TODO generalize to more than 2 explorers
  let btn;
  if (props.idx !== 0) {
    btn = btnClose;
  } else if (!props.comparisonMode && props.allowComparison) {
    btn = btnCompare;
  } else {
    btn = null;
  }

  return (
    <div className="explorer-container">
      <div className="explorer-header">
        {title}
        {btn}
      </div>
      {body}
    </div>
  );
}

export default ExplorerContainer;
