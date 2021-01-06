import React from "react";
import styles from "../App.module.css";
import cx from "../utils/cx";
import Feature from "./Feature";
import Spinner from "./Spinner";
import LabelButton from "./LabelButton";
import Contributions from "./Contributions";
import { VisualizationOutput } from "../models/visualizationOutput";

interface VisualizationProps {
  data: VisualizationOutput;
  instance: number;
  onTargetClick: (
    labelIndex: number,
    inputIndex: number,
    modelIndex: number,
    callback: () => void
  ) => void;
}

interface VisualizationState {
  loading: boolean;
}

class Visualization extends React.Component<
  VisualizationProps,
  VisualizationState
> {
  constructor(props: VisualizationProps) {
    super(props);
    this.state = {
      loading: false,
    };
  }

  onTargetClick = (
    labelIndex: number,
    inputIndex: number,
    modelIndex: number
  ) => {
    this.setState({ loading: true });
    this.props.onTargetClick(labelIndex, inputIndex, modelIndex, () =>
      this.setState({ loading: false })
    );
  };

  //TODO: Refactor the visualization table as a <table> instead of columns, in order to have cleaner styling
  render() {
    const data = this.props.data;
    const isFirstInGroup = this.props.data.model_index === 0;
    const features = data.feature_outputs.map((f) => (
      <Feature data={f} hideHeaders={isFirstInGroup} />
    ));

    return (
      <>
        {this.state.loading && (
          <div className={styles.loading}>
            <Spinner />
          </div>
        )}
        {!isFirstInGroup && <div className={styles["model-separator"]} />}
        <div className={styles["visualization-container"]}>
          <div className={styles["panel__column"]}>
            {isFirstInGroup && (
              <div className={styles["panel__column__title"]}>Predicted</div>
            )}
            <div className={styles["panel__column__body"]}>
              <div className={styles["model-number"]}>
                Model {data.model_index + 1}
              </div>
              {data.predicted.map((p) => (
                <div className={cx([styles.row, styles["row--padding"]])}>
                  <LabelButton
                    onTargetClick={this.onTargetClick}
                    labelIndex={p.index}
                    inputIndex={this.props.instance}
                    modelIndex={this.props.data.model_index}
                    active={p.index === data.active_index}
                  >
                    {p.label} ({p.score.toFixed(3)})
                  </LabelButton>
                </div>
              ))}
            </div>
          </div>
          <div className={styles["panel__column"]}>
            {isFirstInGroup && (
              <div className={styles["panel__column__title"]}>Label</div>
            )}
            <div className={styles["panel__column__body"]}>
              <div className={styles["model-number-spacer"]} />
              <div className={cx([styles.row, styles["row--padding"]])}>
                <LabelButton
                  onTargetClick={this.onTargetClick}
                  labelIndex={data.actual.index}
                  inputIndex={this.props.instance}
                  modelIndex={this.props.data.model_index}
                  active={data.actual.index === data.active_index}
                >
                  {data.actual.label}
                </LabelButton>
              </div>
            </div>
          </div>
          <div className={styles["panel__column"]}>
            {isFirstInGroup && (
              <div className={styles["panel__column__title"]}>Contribution</div>
            )}
            <div className={styles["panel__column__body"]}>
              <div className={styles["model-number-spacer"]} />
              <div className={styles["bar-chart"]}>
                <Contributions feature_outputs={data.feature_outputs} />
              </div>
            </div>
          </div>
          <div
            className={cx([
              styles["panel__column"],
              styles["panel__column--stretch"],
            ])}
          >
            {features}
          </div>
        </div>
      </>
    );
  }
}

export default Visualization;
