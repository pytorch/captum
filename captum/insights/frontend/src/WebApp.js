import React from "react";
import AppBase from "./App";

class WebApp extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      config: [],
      loading: false
    };
    this._fetchInit();
  }

  _fetchInit = () => {
    fetch("/init")
      .then(r => r.json())
      .then(r => this.setState({ config: r }));
  };

  fetchData = filter_config => {
    this.setState({ loading: true });
    fetch("/fetch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(filter_config)
    })
      .then(response => response.json())
      .then(response => this.setState({ data: response, loading: false }));
  };

  onTargetClick = (labelIndex, instance, callback) => {
    fetch("/attribute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ labelIndex, instance })
    })
      .then(response => response.json())
      .then(response => {
        const data = Object.assign([], this.state.data);
        data[instance] = response;
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
