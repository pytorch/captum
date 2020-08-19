import React, { useState } from "react";
import SampleContainer from "./sample/SampleContainer";
import { Tooltip } from "./common/Tooltip";
import { ReactComponent as SwapIcon } from "../icons/swap.svg";
import { ImageLoader } from "./common/ImageLoader";
import { BASE_URL } from "../services/mbx";
import "./Samples.css";

function Sample({ explorer, sampleId }) {
  const explorerId = explorer.id;
  const workspaceId = explorer.workspace.id;
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);

  const onTooltipMount = () => setIsTooltipVisible(true);
  const onTooltipHide = () => setIsTooltipVisible(false);

  const tooltipContent = isTooltipVisible ? (
    <div className="sample-tooltip-content">
      <SampleContainer explorer={explorer} sampleId={sampleId} />
    </div>
  ) : null;

  const params = new URLSearchParams({
    explorer_id: explorerId,
    sample_id: sampleId,
    workspace_id: workspaceId,
  });
  const url = `${BASE_URL}/sample_thumbnail?${params.toString()}`;

  return (
    <div className="sample-thumbnail-container">
      <Tooltip
        content={tooltipContent}
        placement="bottom"
        delayMs={100}
        trigger="mouseenter"
        onMount={onTooltipMount}
        onHide={onTooltipHide}
      >
        <div>
          <ImageLoader
            className="sample-thumbnail"
            alt={`sample [${explorerId}, ${sampleId}]`}
            src={url}
          />
        </div>
      </Tooltip>
    </div>
  );
}

export function SamplesWindow({
  sampleBin,
  anchor = "center",
  columns = 3,
  anchorI = 0,
  useOtherBins = true,
  size = 3,
  ...props
}) {
  const component = sampleBin.component;
  const explorer = component.module.explorer;
  const sampleIds = sampleBin.getSampleWindow(
    anchorI,
    size,
    anchor,
    useOtherBins,
  );
  const samples = sampleIds.map((sampleId, i) => {
    const onSampleClick = () =>
      props.onSampleClick && props.onSampleClick(sampleId);
    return (
      <Sample
        key={i}
        explorer={explorer}
        sampleId={sampleId}
        onClick={onSampleClick}
      />
    );
  });
  columns = Math.min(columns, sampleIds.length);
  const style = {
    gridTemplateColumns: `repeat(${columns}, 1fr)`,
  };
  return (
    <div className="samples-window-grid" style={style}>
      {samples}
    </div>
  );
}

function DistributionBars({ maxHeight = 50, sampleBins }) {
  const values = sampleBins.map(
    (sampleBin) => sampleBin.sortedSampleIds.length,
  );
  const maxValue = Math.max.apply(Math, values);

  return (
    <div className="distribution-bars">
      {values.map((value, i) => {
        const height = Math.floor((value / maxValue) * maxHeight);
        const style = {
          width: `${100 / values.length}%`,
          height: `${height}px`,
        };
        return <div className="distribution-bar" key={i} style={style} />;
      })}
    </div>
  );
}

function Samples({ windowSize = 3, component, ...props }) {
  const [focusMid, setFocusMid] = useState(false);
  if (component == null)
    throw new Error("component object required to display samples");

  const numSampleBins = component.sampleBins.length;
  const [sliderValue, setSliderValue] = useState(
    Math.floor(numSampleBins * 0.5) + 0.5,
  );

  if (numSampleBins === 0) return null;

  const firstSampleBin = component.sampleBins[0];
  const lastSampleBin = component.sampleBins[numSampleBins - 1];

  let sampleBin;
  let sampleIdx;
  if (sliderValue >= numSampleBins) {
    const sampleBinId = numSampleBins - 1;
    sampleBin = component.sampleBins[sampleBinId];
    sampleIdx = sampleBin.sortedSampleIds.length - 1;
  } else {
    const sampleBinId = Math.floor(sliderValue);
    sampleBin = component.sampleBins[sampleBinId];
    sampleIdx = Math.floor(
      (sliderValue - sampleBinId) * sampleBin.sortedSampleIds.length,
    );
  }

  const focusedSize = windowSize * windowSize;
  const focusedColumns = windowSize;
  const unfocusedSize = 4;
  const unfocusedColumns = 2;

  let windowSizeMid, windowSizeLo, columnsMid, columnsLo;

  if (focusMid) {
    windowSizeMid = focusedSize;
    columnsMid = focusedColumns;
    windowSizeLo = unfocusedSize;
    columnsLo = unfocusedColumns;
  } else {
    windowSizeMid = unfocusedSize;
    columnsMid = unfocusedColumns;
    windowSizeLo = focusedSize;
    columnsLo = focusedColumns;
  }

  const windowSizeHi = windowSizeLo;
  const columnsHi = columnsLo;

  const onSliderValueChange = (e) => setSliderValue(parseFloat(e.target.value));

  const onSampleClick = (sampleId) =>
    props.onSampleClick && props.onSampleClick(sampleId);

  const samplesLo = (
    <SamplesWindow
      sampleBin={firstSampleBin}
      anchor="begin"
      onSampleClick={onSampleClick}
      size={windowSizeLo}
      columns={columnsLo}
    />
  );
  const samplesHi = (
    <SamplesWindow
      sampleBin={lastSampleBin}
      anchor="end"
      onSampleClick={onSampleClick}
      size={windowSizeHi}
      columns={columnsHi}
    />
  );
  const samplesMid = (
    <SamplesWindow
      sampleBin={sampleBin}
      anchorI={sampleIdx}
      anchor="center"
      onSampleClick={onSampleClick}
      size={windowSizeMid}
      columns={columnsMid}
      useOtherBins={false}
    />
  );

  const slider = (
    <input
      type="range"
      className="sample-slider"
      min={0}
      max={numSampleBins}
      step={0.01}
      value={sliderValue}
      onChange={onSliderValueChange}
    />
  );

  const onSwapClick = () => setFocusMid(!focusMid);

  const btnSwap = (
    <button onClick={onSwapClick} className="btn-icon">
      <SwapIcon />
    </button>
  );

  return (
    <div className="samples-container">
      <div className="samples-container-top">
        <div className="samples-distribution-slider-container">
          {component.sampleBins == null ? null : (
            <DistributionBars sampleBins={component.sampleBins} />
          )}
          {slider}
          <div>{`Use slider to explore samples across the ${component.type.componentPrefix} axis`}</div>
        </div>
      </div>
      <div className="samples-container-bottom">
        {samplesLo}
        <div className="samples-container-bottom-mid">
          <div className="samples-mid">{samplesMid}</div>
          {btnSwap}
        </div>
        {samplesHi}
      </div>
    </div>
  );
}

export default Samples;
