import React from "react";
import AppBase from "./App";
import { FilterConfig } from "./models/filter";
import { VisualizationGroup } from "./models/visualizationOutput";
import { InsightsConfig } from "./models/insightsConfig";

interface WebAppState {
  data: VisualizationGroup[];
  config: InsightsConfig;
  loading: boolean;
}

class WebApp extends React.Component<{}, WebAppState> {
  constructor(props: {}) {
    super(props);
    this.state = {
      data: [],
      config: {
        classes: [],
        methods: [],
        method_arguments: {},
        selected_method: "",
      },
      loading: false,
    };
    this._fetchInit();
  }

  _fetchInit = () => {
    fetch("init")
      .then((r) => r.json())
      .then((r) => this.setState({ config: r }));
  };

  fetchData = (filter_config: FilterConfig) => {
    this.setState({ loading: true });
    fetch("fetch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(filter_config),
    })
      .then((response) => response.json())
      .then((response) => this.setState({ data: response, loading: false }));
  };

  onTargetClick = (
    labelIndex: number,
    inputIndex: number,
    modelIndex: number,
    callback: () => void
  ) => {
    fetch("attribute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ labelIndex, inputIndex, modelIndex }),
    })
      .then((response) => response.json())
      .then((response) => {
        const data = this.state.data ?? [];
        data[inputIndex][modelIndex] = response;
        this.setState({ data });
        callback();
      });
  };

  render() {
    return (
      <AppBase
        fetchData={this.fetchData}
        fetchInit={this._fetchInit}
        onTargetClick={this.onTargetClick}
        data={this.state.data}
        config={this.state.config}
        loading={this.state.loading}
      />
    );
  }
}

export default WebApp;
