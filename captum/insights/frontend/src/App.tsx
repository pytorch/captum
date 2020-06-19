import React from "react";
import styles from "./App.module.css";
import Header from "./components/Header";
import cx from "./utils/cx";
import Spinner from "./components/Spinner";
import FilterContainer from "./components/FilterContainer";
import Visualization from "./components/Visualization";
import "./App.css";
import { VisualizationOutput } from "./models/visualizationOutput";
import { FilterConfig } from "./models/filter";

interface VisualizationsProps {
  loading: boolean;
  data: VisualizationOutput[];
  onTargetClick: (labelIndex: number, instance: number, callback: () => void) => void;
}

function Visualizations(props: VisualizationsProps) {
  if (props.loading) {
    return (
      <div className="viz">
        <div className={cx([styles.panel, styles["panel--center"]])}>
          <Spinner />
        </div>
      </div>
    );
  }

  if (!props.data || props.data.length === 0) {
    return (
      <div className={styles.viz}>
        <div className={styles.panel}>
          <div className={styles["panel__column"]}>
            Please press{" "}
            <strong className={styles["text-feature-word"]}>Fetch</strong> to
            start loading data.
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className={styles.viz}>
      {props.data.map((v, i) => (
        <Visualization
          data={v}
          instance={i}
          key={i}
          onTargetClick={props.onTargetClick}
        />
      ))}
    </div>
  );
}

interface AppBaseProps {
  fetchInit: () => void;
  fetchData: (filter_config: FilterConfig) => void;
  config: any;
  data: VisualizationOutput[];
  loading: boolean;
  onTargetClick: (labelIndex: number, instance: number, callback: () => void) => void;
}

class AppBase extends React.Component<AppBaseProps> {
  componentDidMount() {
    this.props.fetchInit();
  }

  render() {
    return (
      <div className={styles.app}>
        <Header />
        <FilterContainer
          fetchData={this.props.fetchData}
          config={this.props.config}
          key={this.props.config.classes}
        />
        <Visualizations
          data={this.props.data}
          loading={this.props.loading}
          onTargetClick={this.props.onTargetClick}
        />
      </div>
    );
  }
}

export default AppBase;
