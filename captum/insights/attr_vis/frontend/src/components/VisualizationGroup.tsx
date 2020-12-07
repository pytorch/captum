import React from "react";
import styles from "../App.module.css";
import cx from "../utils/cx";
import Visualization from "../components/Visualization";
import { VisualizationGroup } from "../models/visualizationOutput";

interface VisualizationGroupDisplayProps {
  inputIndex: number;
  data: VisualizationGroup;
  onTargetClick: (
    labelIndex: number,
    inputIndex: number,
    modelIndex: number,
    callback: () => void
  ) => void;
}

function VisualizationGroupDisplay(props: VisualizationGroupDisplayProps) {
  return (
    <div
      className={cx({
        [styles.panel]: true,
        [styles["panel--long"]]: true,
      })}
    >
      {props.data.map((v, i) => (
        <Visualization
          data={v}
          instance={props.inputIndex}
          onTargetClick={props.onTargetClick}
          key={i}
        />
      ))}
    </div>
  );
}

export default VisualizationGroupDisplay;
