import React from "react";
import ReactDOM from "react-dom";
import AppBase from "../../src/App";
import * as widgets from "@jupyter-widgets/base";
import * as _ from "lodash";

class Widget extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      config: {
        classes: [],
        methods: [],
        method_arguments: {},
      },
      loading: false,
      callback: null,
    };
    this.backbone = this.props.backbone;
  }

  componentDidMount() {
    this.backbone.model.on("change:output", this._outputChanged, this);
    this.backbone.model.on(
      "change:attribution",
      this._attributionChanged,
      this
    );
  }

  _outputChanged(model, output, options) {
    if (_.isEmpty(output)) return;
    this.setState({ data: output, loading: false });
  }

  _attributionChanged(model, attribution, options) {
    if (_.isEmpty(attribution)) return;
    const data = Object.assign([], this.state.data);
    const callback = this.state.callback;
    const labelDetails = model.attributes.label_details;
    data[labelDetails.inputIndex][labelDetails.modelIndex] = attribution;
    this.setState({ data: data, callback: null }, () => {
      callback();
    });
  }

  _fetchInit = () => {
    this.setState({
      config: this.backbone.model.get("insights_config"),
    });
  };

  fetchData = (filterConfig) => {
    this.setState({ loading: true }, () => {
      this.backbone.model.save({ config: filterConfig, output: [] });
    });
  };

  onTargetClick = (labelIndex, inputIndex, modelIndex, callback) => {
    this.setState({ callback: callback }, () => {
      this.backbone.model.save({
        label_details: { labelIndex, inputIndex, modelIndex },
        attribution: {},
      });
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

var CaptumInsightsModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: "CaptumInsightsModel",
    _view_name: "CaptumInsightsView",
    _model_module: "jupyter-captum-insights",
    _view_module: "jupyter-captum-insights",
    _model_module_version: "0.1.0",
    _view_module_version: "0.1.0",
  }),
});

var CaptumInsightsView = widgets.DOMWidgetView.extend({
  initialize() {
    const $app = document.createElement("div");
    ReactDOM.render(<Widget backbone={this} />, $app);
    this.el.append($app);
  },
});

export { Widget as default, CaptumInsightsModel, CaptumInsightsView };
