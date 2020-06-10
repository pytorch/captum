import React from "react";
import Filter from "./Filter";

function parseEventTargetValue(target) {
  switch (target.type) {
    case "checkbox":
      return target.checked;
    case "number":
      return parseInt(target.value);
    default:
      return target.value;
  }
}

class FilterContainer extends React.Component {
  constructor(props) {
    super(props);
    const suggested_classes = props.config.classes.map((c, i) => ({
      id: i,
      name: c,
    }));
    this.state = {
      prediction: "all",
      classes: [],
      suggested_classes: suggested_classes,
      selected_method: props.config.selected_method,
      method_arguments: props.config.method_arguments,
    };
  }

  handleClassDelete = (i) => {
    const classes = this.state.classes.slice(0);
    const removed_class = classes.splice(i, 1);
    const suggested_classes = [].concat(
      this.state.suggested_classes,
      removed_class
    );
    this.setState({ classes, suggested_classes });
  };

  handleClassAdd = (added_class) => {
    const classes = [].concat(this.state.classes, added_class);
    const suggested_classes = this.state.suggested_classes.filter(
      (t) => t.id !== added_class.id
    );
    this.setState({ classes, suggested_classes });
  };

  handleInputChange = (event) => {
    const target = event.target;
    const value = parseEventTargetValue(event.target);
    const name = target.name;
    this.setState({
      [name]: value,
    });
  };

  handleArgumentChange = (event) => {
    const target = event.target;
    const name = target.name;
    const value = parseEventTargetValue(target);
    const method_arguments = this.state.method_arguments;
    method_arguments[this.state.selected_method][name].value = value;
    this.setState({ method_arguments });
  };

  handleSubmit = (event) => {
    const method = this.state.selected_method;
    const method_arguments = this.state.method_arguments;
    const argument_config =
      method in method_arguments ? method_arguments[method] : {};
    const args = {};
    Object.keys(argument_config).forEach(function (key) {
      args[key] = argument_config[key].value;
    });
    const data = {
      prediction: this.state.prediction,
      classes: this.state.classes.map((i) => i["name"]),
      attribution_method: method,
      arguments: args,
    };
    this.props.fetchData(data);
    event.preventDefault();
  };

  render() {
    return (
      <Filter
        prediction={this.state.prediction}
        classes={this.state.classes}
        suggestedClasses={this.state.suggested_classes}
        selectedMethod={this.state.selected_method}
        methodArguments={this.state.method_arguments}
        methods={this.props.config.methods}
        handleClassAdd={this.handleClassAdd}
        handleClassDelete={this.handleClassDelete}
        handleInputChange={this.handleInputChange}
        handleArgumentChange={this.handleArgumentChange}
        handleSubmit={this.handleSubmit}
      />
    );
  }
}

export default FilterContainer;
