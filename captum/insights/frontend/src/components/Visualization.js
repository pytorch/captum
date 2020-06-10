import React from "react";
import styles from "../App.module.css";
import cx from "../utils/cx";
import Feature from "./Feature";
import Spinner from "./Spinner";
import LabelButton from "./LabelButton";
import Contributions from "./Contributions";

class Visualization extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: false,
    };
  }

  onTargetClick = (labelIndex, instance) => {
    this.setState({ loading: true });
    this.props.onTargetClick(labelIndex, instance, () =>
      this.setState({ loading: false })
    );
  };

  render() {
    const data = this.props.data;
    console.log("visualization", data);
    const features = data.feature_outputs.map((f) => <Feature data={f} />);

    return (
      <>
        {this.state.loading && (
          <div className={styles.loading}>
            <Spinner />
          </div>
        )}
        <div
          className={cx({
            [styles.panel]: true,
            [styles["panel--long"]]: true,
            [styles["panel--loading"]]: this.state.loading,
          })}
        >
          <div className={styles["panel__column"]}>
            <div className={styles["panel__column__title"]}>Predicted</div>
            <div className={styles["panel__column__body"]}>
              {data.predicted.map((p) => (
                <div className={cx([styles.row, styles["row--padding"]])}>
                  <LabelButton
                    onTargetClick={this.onTargetClick}
                    labelIndex={p.index}
                    instance={this.props.instance}
                    active={p.index === data.active_index}
                  >
                    {p.label} ({p.score.toFixed(3)})
                  </LabelButton>
                </div>
              ))}
            </div>
          </div>
          <div className={styles["panel__column"]}>
            <div className={styles["panel__column__title"]}>Label</div>
            <div className={styles["panel__column__body"]}>
              <div className={cx([styles.row, styles["row--padding"]])}>
                <LabelButton
                  onTargetClick={this.onTargetClick}
                  labelIndex={data.actual.index}
                  instance={this.props.instance}
                  active={data.actual.index === data.active_index}
                >
                  {data.actual.label}
                </LabelButton>
              </div>
            </div>
          </div>
          <div className={styles["panel__column"]}>
            <div className={styles["panel__column__title"]}>Contribution</div>
            <div className={styles["panel__column__body"]}>
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
